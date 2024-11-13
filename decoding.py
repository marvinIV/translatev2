from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Categorical

from .audio import CHUNK_LENGTH
from .tokenizer import Tokenizer, get_tokenizer
from .utils import compression_ratio

if TYPE_CHECKING:
    from .model import Whisper

import kenlm
import time
@torch.no_grad()
def detect_language(
    model: "Whisper", mel: Tensor, tokenizer: Tokenizer = None
) -> Tuple[Tensor, List[dict]]:
    """
    Detect the spoken language in the audio, and return them as list of strings, along with the ids
    of the most probable language tokens and the probability distribution over all language tokens.
    This is performed outside the main decode loop in order to not interfere with kv-caching.

    Returns
    -------
    language_tokens : Tensor, shape = (n_audio,)
        ids of the most probable language tokens, which appears after the startoftranscript token.
    language_probs : List[Dict[str, float]], length = n_audio
        list of dictionaries containing the probability distribution over all languages.
    """
    if tokenizer is None:
        tokenizer = get_tokenizer(
            model.is_multilingual, num_languages=model.num_languages
        )
    if (
        tokenizer.language is None
        or tokenizer.language_token not in tokenizer.sot_sequence
    ):
        raise ValueError(
            "This model doesn't have language tokens so it can't perform lang id"
        )

    single = mel.ndim == 2
    if single:
        mel = mel.unsqueeze(0)

    # skip encoder forward pass if already-encoded audio features were given
    if mel.shape[-2:] != (model.dims.n_audio_ctx, model.dims.n_audio_state):
        mel = model.encoder(mel)

    # forward pass using a single token, startoftranscript
    n_audio = mel.shape[0]
    x = torch.tensor([[tokenizer.sot]] * n_audio).to(mel.device)  # [n_audio, 1]
    logits = model.logits(x, mel)[:, 0]

    # collect detected languages; suppress all non-language tokens
    mask = torch.ones(logits.shape[-1], dtype=torch.bool)
    mask[list(tokenizer.all_language_tokens)] = False
    logits[:, mask] = -np.inf
    language_tokens = logits.argmax(dim=-1)
    language_token_probs = logits.softmax(dim=-1).cpu()
    language_probs = [
        {
            c: language_token_probs[i, j].item()
            for j, c in zip(tokenizer.all_language_tokens, tokenizer.all_language_codes)
        }
        for i in range(n_audio)
    ]

    if single:
        language_tokens = language_tokens[0]
        language_probs = language_probs[0]

    return language_tokens, language_probs


@dataclass(frozen=True)
class DecodingOptions:
    task: str = "translate"  # whether to perform X->X "transcribe" or X->English "translate"
    language: Optional[str] = None  # language that the audio is in; uses detected language if None

    # sampling-related options
    temperature: float = 0.0
    sample_len: Optional[int] = None  # maximum number of tokens to sample
    best_of: Optional[int] = None     # number of independent samples to collect, when t > 0
    beam_size: Optional[int] = None   # number of beams in beam search, when t == 0
    patience: Optional[float] = None  # patience in beam search (https://arxiv.org/abs/2204.05424)

    # options for ranking generations (either beams or best-of-N samples)
    length_penalty: Optional[float] = None   # "alpha" in Google NMT, None defaults to length norm

    # prompt, prefix, and token suppression
    prompt: Optional[Union[str, List[int]]] = None   # text or tokens for the previous context
    prefix: Optional[Union[str, List[int]]] = None   # text or tokens to prefix the current context
    suppress_blank: bool = True                      # this will suppress blank outputs

    # list of tokens ids (or comma-separated token ids) to suppress
    # "-1" will suppress a set of symbols as defined in `tokenizer.non_speech_tokens()`
    suppress_tokens: Optional[Union[str, Iterable[int]]] = "-1"

    # timestamp sampling options
    without_timestamps: bool = False              # use <|notimestamps|> to sample text tokens only
    max_initial_timestamp: Optional[float] = 1.0  # the initial timestamp cannot be later than this

    # implementation details
    fp16: bool = True  # use fp16 for most of the calculation

    #with LM
    withlm: bool = False
    lm_path: str = None
    lm_alpha: float = 2.0
    lm_beta: float = 2.5
    
    #debug
    debug: bool = False
    #select candidates
    select_candidates: int = None



@dataclass(frozen=True)
class DecodingResult:
    audio_features: Tensor
    language: str
    language_probs: Optional[Dict[str, float]] = None
    tokens: List[int] = field(default_factory=list)
    text: str = ""
    avg_logprob: float = np.nan
    no_speech_prob: float = np.nan
    temperature: float = np.nan
    compression_ratio: float = np.nan


class Inference:
    def logits(self, tokens: Tensor, audio_features: Tensor) -> Tensor:
        """Perform a forward pass on the decoder and return per-token logits"""
        raise NotImplementedError

    def rearrange_kv_cache(self, source_indices) -> None:
        """Update the key-value cache according to the updated beams"""
        raise NotImplementedError

    def cleanup_caching(self) -> None:
        """Clean up any resources or hooks after decoding is finished"""
        pass


class PyTorchInference(Inference):
    def __init__(self, model: "Whisper", initial_token_length: int):
        self.model: "Whisper" = model
        self.initial_token_length = initial_token_length
        self.kv_cache = {}
        self.hooks = []

        key_modules = [block.attn.key for block in self.model.decoder.blocks]
        value_modules = [block.attn.value for block in self.model.decoder.blocks]
        self.kv_modules = key_modules + value_modules

    def logits(self, tokens: Tensor, audio_features: Tensor) -> Tensor:
        if not self.kv_cache:
            self.kv_cache, self.hooks = self.model.install_kv_cache_hooks()

        if tokens.shape[-1] > self.initial_token_length:
            # only need to use the last token except in the first forward pass
            tokens = tokens[:, -1:]

        return self.model.decoder(tokens, audio_features, kv_cache=self.kv_cache)

    def cleanup_caching(self):
        for hook in self.hooks:
            hook.remove()

        self.kv_cache = {}
        self.hooks = []

    def rearrange_kv_cache(self, source_indices):
        if source_indices != list(range(len(source_indices))):
            for module in self.kv_modules:
                # update the key/value cache to contain the selected sequences
                self.kv_cache[module] = self.kv_cache[module][source_indices].detach()


class SequenceRanker:
    def rank(
        self, tokens: List[List[Tensor]], sum_logprobs: List[List[float]]
    ) -> List[int]:
        """
        Given a list of groups of samples and their cumulative log probabilities,
        return the indices of the samples in each group to select as the final result
        """
        raise NotImplementedError


class MaximumLikelihoodRanker(SequenceRanker):
    """
    Select the sample with the highest log probabilities, penalized using either
    a simple length normalization or Google NMT paper's length penalty
    """

    def __init__(self, length_penalty: Optional[float]):
        self.length_penalty = length_penalty

    def rank(self, tokens: List[List[Tensor]], sum_logprobs: List[List[float]]):
        def scores(logprobs, lengths):
            result = []
            for logprob, length in zip(logprobs, lengths):
                if self.length_penalty is None:
                    penalty = length
                else:
                    # from the Google NMT paper
                    penalty = ((5 + length) / 6) ** self.length_penalty
                result.append(logprob / penalty)
            return result

        # get the sequence with the highest score
        lengths = [[len(t) for t in s] for s in tokens]
        return [np.argmax(scores(p, l)) for p, l in zip(sum_logprobs, lengths)]


class TokenDecoder:
    def reset(self):
        """Initialize any stateful variables for decoding a new sequence"""

    def update(
        self, tokens: Tensor, logits: Tensor, sum_logprobs: Tensor
    ) -> Tuple[Tensor, bool]:
        """Specify how to select the next token, based on the current trace and logits

        Parameters
        ----------
        tokens : Tensor, shape = (n_batch, current_sequence_length)
            all tokens in the context so far, including the prefix and sot_sequence tokens

        logits : Tensor, shape = (n_batch, vocab_size)
            per-token logits of the probability distribution at the current step

        sum_logprobs : Tensor, shape = (n_batch)
            cumulative log probabilities for each sequence

        Returns
        -------
        tokens : Tensor, shape = (n_batch, current_sequence_length + 1)
            the tokens, appended with the selected next token

        completed : bool
            True if all sequences has reached the end of text

        """
        raise NotImplementedError

    def finalize(
        self, tokens: Tensor, sum_logprobs: Tensor
    ) -> Tuple[Sequence[Sequence[Tensor]], List[List[float]]]:
        """Finalize search and return the final candidate sequences

        Parameters
        ----------
        tokens : Tensor, shape = (n_audio, n_group, current_sequence_length)
            all tokens in the context so far, including the prefix and sot_sequence

        sum_logprobs : Tensor, shape = (n_audio, n_group)
            cumulative log probabilities for each sequence

        Returns
        -------
        tokens : Sequence[Sequence[Tensor]], length = n_audio
            sequence of Tensors containing candidate token sequences, for each audio input

        sum_logprobs : List[List[float]], length = n_audio
            sequence of cumulative log probabilities corresponding to the above

        """
        raise NotImplementedError


class GreedyDecoder(TokenDecoder):
    def __init__(self, temperature: float, eot: int):
        self.temperature = temperature
        self.eot = eot

    def update(
        self, tokens: Tensor, logits: Tensor, sum_logprobs: Tensor
    ) -> Tuple[Tensor, bool]:
        print("USING GREEDY DECODER LM")
        if self.temperature == 0:
            next_tokens = logits.argmax(dim=-1)
        else:
            next_tokens = Categorical(logits=logits / self.temperature).sample()

        logprobs = F.log_softmax(logits.float(), dim=-1)
        current_logprobs = logprobs[torch.arange(logprobs.shape[0]), next_tokens]
        sum_logprobs += current_logprobs * (tokens[:, -1] != self.eot)

        next_tokens[tokens[:, -1] == self.eot] = self.eot
        tokens = torch.cat([tokens, next_tokens[:, None]], dim=-1)

        completed = (tokens[:, -1] == self.eot).all()
        return tokens, completed

    def finalize(self, tokens: Tensor, sum_logprobs: Tensor):
        # make sure each sequence has at least one EOT token at the end
        tokens = F.pad(tokens, (0, 1), value=self.eot)
        return tokens, sum_logprobs.tolist()

class BeamSearchDecoderWithLM(TokenDecoder):
    def __init__(self, beam_size: int, eot: int, inference: Inference, patience: Optional[float] = None, 
                    lm_path: Optional[str] = None, lm_alpha: Optional[float] = 0.5, lm_beta: Optional[float] = 0.5, select_candidates: Optional[int] =100):
        self.beam_size = beam_size
        self.eot = eot
        self.inference = inference
        self.patience = patience or 1.0
        self.max_candidates: int = round(beam_size * self.patience)
        self.finished_sequences = None
        self.lm: "kenlm.Model" = kenlm.Model(lm_path) if lm_path else None
        self.tokenizer = get_tokenizer('en')
        self.lm_alpha = lm_alpha
        self.lm_beta = lm_beta
        self.select_candidates = select_candidates
        with open("/home/marvin.rajwadi@CHASEITS/KenLM/juju_test/engchar.txt", 'r') as file:
            # Read each line from the file and append to the list
            data_list = file.readlines()
            print("datalist loaded")

        # Strip newline characters from each element
        self.data_list = [int(item.strip()) for item in data_list]

        assert self.max_candidates > 0, f"Invalid beam size ({beam_size}) or patience ({patience})"
   
    def reset(self):
        self.finished_sequences = None

    

    def update(self, tokens: Tensor, logits: Tensor, sum_logprobs: Tensor) -> Tuple[Tensor, bool]:
        print("USING BEAM DECODER LM")
        if tokens.shape[0] % self.beam_size != 0:
            raise ValueError(f"{tokens.shape}[0] % {self.beam_size} != 0")

        n_audio = tokens.shape[0] // self.beam_size
        print(f"n_audio: {n_audio}, tokens.shape: {tokens.shape}, self.beam_size: {self.beam_size}")
        #print([self.tokenizer.decode(x)[3:] for x in tokens])
        if self.finished_sequences is None:  # for the first update
            self.finished_sequences = [{} for _ in range(n_audio)]

        logprobs = F.log_softmax(logits.float(), dim=-1)
        # https://github.com/openai/whisper/discussions/361
        # we only use tokenizer token, all token id after 50364 and 50364 are all timestamp tokens
        # because we pass without_timestamp=True to tokenizer, we only get token id < 50364
        logprobs_for_lm = F.log_softmax(logits.float(), dim=-1)#[:, :50364 + 1]
        
        # def hotspot(prefix_,next_token_):
        #     score_=0
        #     #key_w=['交易日', '本幣', '台幣', '瑞士法郎', '擔保', '韓元', '交割日', '金額', '新台幣', '即期', '換匯', '美金', '外幣', '利率', '核對', 'spot', '人民幣', '港幣', '加幣', '股價', '日圓', '價位', '匯率', '歐元', '央行', '匯價', '南非幣', '澳幣', '短期', '新幣', '交易', '信用', '交換', '交割', '美元', '外匯']
            
        #     key_w = [" Aine"," AMT"," Anchor"," Asir"," Boipatong"," Boucher"," Bowie"," Boyle"," Cahill"," Cam"," Ciaran"," clause"," Courtney"," dinah"," Eckman"," Eliza"," Erick"," Ernests"," Femina"," Geoff"," gers"," HD"," Hogerheide"," Hogwarts"," Jacques"," junkie"," louse"," Mccallum"," Meghan"," Moleko"," MPO", " Mullally", " Navbharat", " Oireachtas", " Oscar", " Ronan", " RTE", " SCC", " Vanessa", " Vicky", " zingermans",""]

        #     if True:
        #         for ix,ip in enumerate(key_w):
        #             weight = [5]*len(key_w)
        #             ip = self.tokenizer.encode(ip)
        #             if next_token_ in ip:
        #                 cur_idx=ip.index(next_token_)
        #                 #print(cur_idx)
        #                 min_len = len(ip)
        #                 if cur_idx == 0:
        #                     # scale score by length of unigram matched so far
        #                     print("BOOSTING___________________________________")
        #                     print(f"Current token: {self.tokenizer.decode([next_token_])}")
        #                     score_ = self.lm.BaseScore(last_state, chr(next_token_ + 100), new_token_state)
        #                     score_ = self.lm_alpha*score_ + weight[ix] * (cur_idx+1)/min_len
        #                     # if weight[ix] <0:
        #                     #     score_ = self.lm_alpha*score_ + weight[ix] #* cur_idx+1/min_len
                            
        #                     #return score_
        #                 else:
        #                    # print(prefix[-1*(cur_idx):] )
        #                     #print(i[:cur_idx])
        #                     if prefix_[-1*(cur_idx):] == ip[:cur_idx]:
        #                         print("BOOSTING___________________________________")
        #                         print(f"Current token: {self.tokenizer.decode([next_token_])}")
        #                         score_ = self.lm.BaseScore(last_state, chr(next_token_ + 100), new_token_state)
        #                         score_ = self.lm_alpha*score_ + weight[ix] * (cur_idx+1)/min_len
                                
                        
                    
        #         #return score_
        #         return 0
        # # List of tokens to boost
        # # boost_tokens = [" Aine"," AMT"," Anchor"," Asir"," Boipatong"," Boucher"," Bowie"," Boyle"," Cahill"," Cam"," Ciaran"," clause"," Courtney"," dinah"," Eckman"," Eliza"," Erick"," Ernests"," Femina"," Geoff"," gers"," HD"," Hogerheide"," Hogwarts"," Jacques"," junkie"," louse"," Mccallum"," Meghan"," Moleko"," MPO", " Mullally", " Navbharat", " Oireachtas", " Oscar", " Ronan", " RTE", " SCC", " Vanessa", " Vicky", " zingermans"]

        # boost_tokens = ['Linda']
        # boost_tokens = [self.tokenizer.encode(x) for x in boost_tokens]
        # boost_tokens = [item for sublist in boost_tokens for item in sublist]
        # # Function to calculate final score for tokens
        # def calculate_final_scores(whisper_score, lm_score_, c_token):
        
        #     # for token in tokens:
        #     #     whisper_score = whisper_scores[token]
        #     #     kenlm_score = kenlm_scores[token]
        
        #     # Check if token is in the boost list and has a whisper score of -inf
        #     if c_token in boost_tokens:# and whisper_score == float('-inf'):
        #         # Check if KenLM score is greater than -0.5
        #         print("-----------------------",self.tokenizer.decode([c_token]))
                
        #         kenlm_score = self.lm_alpha * self.lm.BaseScore(last_state, chr(c_token + 100), new_token_state)
        #         print("-----------------------", kenlm_score, whisper_score)
        #         if kenlm_score > -0.2:# and whisper_score < kenlm_score:# check higher whisper score
        #             final_score = kenlm_score
        #         else:
        #             final_score = whisper_score + kenlm_score
        #         #print(f"whisper: {whisper_score}  Kenlm: {kenlm_score} boost: {final_score}")
        #     else:
        #         final_score = whisper_score #+  lm_score_
        #         #print(f"whisper: {whisper_score}  Kenlm: {kenlm_score} boost: {final_score}")
            
           
        
        #     return final_score

    #     def check_token(x):
    #         english_chars = [
    #     'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
    #     'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
    #     'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
    #     'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    #     '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',#' ',
    #     ',', '.', '!', '?', ';', ':', "'", '"', '-', '(', ')', '[', ']', '{', '}'
    # ]
    #         # Dictionary to hold non-English characters and their IDs
    #         english_char_ids = []
    #         # Loop through each ID in the tokenizer's vocabulary
    #         # for i in range(50364):
    #         char = tokenizer.decode([x])
    #         #print(char)
    #         # Check if the character is non-English by seeing if it's in the English character set
    #         for k in english_chars:
    #             #print(k)
    #             if k in char:
    #                 return True
    #             else:
    #                 t=False
    #         return False
        # def check_token(x):
        #     english_chars = set(
        #         'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789' +
        #         ',.!?\';:"-()[]{}'
        #     )
        #     # english_chars = set(
        #     #     'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzÀÂÆÇÈÉÊËÎÏÔŒÙÛÜàâæçèéêëîïôœùûüÿ' +
        #     #     ',.!?\';:"-()[]{}'
        #     # )
        #     char = self.tokenizer.decode([x])
        #     # Return True only if all characters in the token are in the English set
        #     return all(c in english_chars for c in char)
        
        # # Example usage
        # tokens = ["example", "tokens", "here"]
        # whisper_scores = {"example": -1.0, "tokens": float('-inf'), "here": -0.5}
        # kenlm_scores = {"example": -0.2, "tokens": -0.3, "here": -0.1}
        # lm_alpha = 0.5
        
        # final_scores = calculate_final_scores(whisper_scores, kenlm_scores, lm_aplha)
        # print(final_scores)

                            
        #print(self.tokenizer.decode(tokens[0]))
        #sort logprobs_for_lm in descending order and get values
        # if self.select_candidates is not None:
        #     new_logprobs_for_lm=[]
        #     for i in logprobs_for_lm:
        #         values, indices = torch.sort(i, descending=True)
        #         #print([(self.tokenizer.decode([x]),y) for x,y in zip(indices.tolist()[:5],values.tolist()[:5])])
        #         #select the top n candidates based on whisper log prob
        #         filtered_indices = []
        #         for idx in indices:
        #             if check_token(idx.item()) and len(filtered_indices) < self.select_candidates:
        #                 filtered_indices.append(idx.item())
            
        #         new_logprobs_for_lm.append(filtered_indices)
                #new_logprobs_for_lm.append(indices.tolist()[:self.select_candidates])
                # new_logprobs_for_lm.append(filtered_indices)
            #merge all top 5 logs to get all possible candidate tokens
            #candidates = np.unique(new_logprobs_for_lm[0]+new_logprobs_for_lm[1]+new_logprobs_for_lm[2]+new_logprobs_for_lm[3]+new_logprobs_for_lm[4])
            # candidates = new_logprobs_for_lm
        
        next_tokens, source_indices, finished_sequences = [], [], []
        #print("n_audio: ",n_audio)
        #start time 
        for i in range(int(n_audio)):
            #print("start")
            start = time.time()
            scores, sources, finished = {}, {}, {}

            # STEP 1: calculate the cumulative log probabilities for possible candidates
            for j in range(self.beam_size):
                idx = i * self.beam_size + j
                prefix = tokens[idx].tolist()
                skip_len=prefix.index(50258)
                if skip_len <0:
                    skip_len = 0
                #skip_len = skip_len+4
                skip_len=4
                print(self.tokenizer.decode(prefix))
                #print(skip_len)
                #break
                lm_score = [float("-inf")]*4
                # we skip first 4 special tokens
                if (len(prefix) > skip_len):
                    curr_state = kenlm.State()
                    next_state = kenlm.State()
                    self.lm.BeginSentenceWrite(curr_state)
                    # lm_score = []
                    prob = 0.0
                    for k in range(len(prefix[skip_len:])):
                        prob += self.lm.BaseScore(curr_state, chr(prefix[k + skip_len] + 100), next_state)
                        curr_state, next_state = next_state, curr_state
                    # save last state so that we do not have to recompute the whole sentence
                    last_state = curr_state
                    
                    # calculate all log10 probabilities of all tokens
                    # this is not efficient: https://github.com/kpu/kenlm/issues/367
                #     if self.select_candidates is not None:
                #         print([self.tokenizer.decode([x]) for x in candidates[idx]])
                #         for k in candidates[idx]:
                #             new_token_state = kenlm.State()
                #             #new_token_score = self.lm.BaseScore(last_state, chr(k + 100), new_token_state)
                #             #boost hotwords
                #             # lm_hw = hotspot(prefix,k)
                #             # #logprobs_for_lm[idx][k] = logprobs_for_lm[idx][k]+ self.lm_alpha*new_token_score + self.lm_beta
                #             # if lm_hw != 0:
                #             #     print("++++++++++++++++")
                #             #     logprobs_for_lm[idx][k] =logprobs_for_lm[idx][k] + lm_hw
                #             #     #logprobs_for_lm[idx][k] =lm_hw
                                
                #             # # else:
                #             # #     logprobs_for_lm[idx][k] =logprobs_for_lm[idx][k] + self.lm_alpha * self.lm.BaseScore(last_state, chr(k + 100), new_token_state)
                #             # else:
                #             # if True:
                #             #lm_hw = self.lm_alpha * self.lm.BaseScore(last_state, chr(k + 100), new_token_state)
                #             #logprobs_for_lm[idx][k] =logprobs_for_lm[idx][k]+lm_hw
                #             # print(lm_hw)

                            
                #             # lm_hw= -1.9
                #             # #logprobs_for_lm[idx][k] =logprobs_for_lm[idx][k]+lm_hw
                #             # before= logprobs[idx][k]
                #             # new_score = calculate_final_scores(logprobs_for_lm[idx][k],lm_hw,k)
                #             # logprobs_for_lm[idx][k] = new_score
                            
                            
                #             #print(f"kenlm new token score: {self.tokenizer.decode([k])} : {lm_hw}  Shallow fusion : {logprobs_for_lm[idx][k] } whisper: {before}  score: {new_score}")
                        
                #             # else:
                #             #     lm_hw=float("-inf")
                #             #lm_score.append(lm_hw)
                #             #lm_score.append(float("-inf"))
                #             #print(f"kenlm new token score: {self.tokenizer.decode([k])} : {new_token_score}")
                #         #print(f"kenlm new token score: {self.tokenizer.decode([np.argmax(lm_score)])}")

                #         ###################################
                        
                #         #lm_score.append(self.lm.BaseScore(last_state, "</s>", new_token_state))
                
                #         ##############################################################################
                        
                        


                #         # new_score = [float('-inf')]*50365
                        
                #         # for idc,c in enumerate(candidates[j]):
                #         #     # assign index values of lm score to new_score
                #         #     new_score[c] = lm_score[idc]
                #         #     #logprobs_for_lm[idx][c] = self.lm_alpha * lm_score[idc]+ (1 - self.lm_alpha) * logprobs_for_lm[idx][c]
                #         #     #logprobs_for_lm[idx][c] = lm_score[idc]+logprobs_for_lm[idx][c]
                #         # #new_score[50257] = self.lm.BaseScore(last_state, "</s>", new_token_state)
                #         # lm_score = new_score
                #         # lm_score = torch.FloatTensor(lm_score).to(tokens.device)

                #     else:
                #         for k in range(50364):
                #             new_token_state = kenlm.State()
                #             new_token_score = self.lm.BaseScore(last_state, chr(k + 100), new_token_state)
                #             lm_score.append(new_token_score)
                #         # print(f"kenlm new token score: {self.tokenizer.decode(np.argmax(new_token_score))}")
                #         # print(self.tokenizer.decode(prefix))
                #         # add <endoftext> token's probability, which is </s> in kenlm
                #         #lm_score.append(self.lm.BaseScore(last_state, "</s>", new_token_state))
                #         lm_score = torch.FloatTensor(lm_score).to(tokens.device)

                    

                    
                #     # common shallow fusion
                #     #temp_logprobs_idx = logprobs_for_lm[idx] + self.lm_alpha*lm_score#+5.5
                #     temp_logprobs_idx=logprobs_for_lm[idx]#+ self.lm_alpha*lm_score + self.lm_beta#*len(prefix)
                #     # print("################################################################")
                #     # print(self.tokenizer.decode(temp_logprobs_idx.topk(self.beam_size + 1)[1]))
                #     # print(len(temp_logprobs_idx.topk(self.beam_size + 1)))
                #     # temp_logprobs_idx = self.lm_alpha*lm_score
                # else:
                #     temp_logprobs_idx = logprobs_for_lm[idx]

                # def get_top_k_elements_with_indices_np(data, k):
                #             # Convert data to a numpy array
                #             data_array = np.array(data)
                #             # Get the indices of the top k elements in descending order
                #             top_k_indices = np.argsort(-data_array)[:k]
                #             # Extract the top k values using these indices
                #             top_k_values = data_array[top_k_indices]
                #             # Return a list of tuples containing indices and values
                #             return list(zip(top_k_indices, top_k_values))
                        
                # top_kc = get_top_k_elements_with_indices_np(lm_score, 5)
                        
                # Display the result
                #print("Top 3 elements with indices:")
                # for index, value in top_kc:
                #     #print(f"Index: {index}, Value: {value}")
                #     print(f"kenlm new token score: {self.tokenizer.decode([candidates[idx][index]])} : {value}")
                # ###############################################################################
                # # create a empty list to mtch whisper lm shape of 50365
                # # new_score=[0.0]*50365

                for logprob, token in zip(*logprobs_for_lm[idx].topk(self.beam_size + 1)):
                    #print("################################################################")
                    print(f"whisper top k: {self.tokenizer.decode([token])} {logprob}")
                    logprob = (sum_logprobs[idx] + logprob).item() 
                    sequence = tuple(prefix + [token.item()])
                    scores[sequence] = logprob
                    sources[sequence] = idx
            #print(f"Scores: {scores}")
            #print(f"Sources: {sources}")
            # STEP 2: rank the candidates and keep the top beam_size sequences for each audio
            saved = 0
            for sequence in sorted(scores, key=scores.get, reverse=True):
                print(self.tokenizer.decode(sequence), scores[sequence])
                if sequence[-1] == self.eot:
                    finished[sequence] = scores[sequence]
                else:
                    sum_logprobs[len(next_tokens)] = scores[sequence]
                    next_tokens.append(sequence)
                    source_indices.append(sources[sequence])

                    saved += 1
                    if saved == self.beam_size:
                        break

            finished_sequences.append(finished)
            stop = time.time()
            print("time for token gen beamdecoderLM: ",stop-start)
        #stop time
        tokens = torch.tensor(next_tokens, device=tokens.device)
        #for x_o in tokens:
            #print(f"finished: {self.tokenizer.decode(x_o)}")
        print("---------------------------------------------")
        self.inference.rearrange_kv_cache(source_indices)

        # add newly finished sequences to self.finished_sequences
        assert len(self.finished_sequences) == len(finished_sequences)
        for previously_finished, newly_finished in zip(self.finished_sequences, finished_sequences):
            print("prev fin: ",[self.tokenizer.decode(g) for g in previously_finished])
            for seq in sorted(newly_finished, key=newly_finished.get, reverse=True):
                print("fin: ",self.tokenizer.decode(seq))
                if len(previously_finished) >= self.max_candidates:
                    break  # the candidate list is full
                previously_finished[seq] = newly_finished[seq]

        # mark as completed if all audio has enough number of samples
        completed = all(
            len(sequences) >= self.max_candidates for sequences in self.finished_sequences
        )
        
        
        return tokens, completed

    def finalize(self, preceding_tokens: Tensor, sum_logprobs: Tensor):
        # collect all finished sequences, including patience, and add unfinished ones if not enough
        sum_logprobs = sum_logprobs.cpu()
        for i, sequences in enumerate(self.finished_sequences):
            #print("finished: ", sequences)
            if len(sequences) < self.beam_size:  # when not enough sequences are finished
                for j in list(np.argsort(sum_logprobs[i]))[::-1]:
                    sequence = preceding_tokens[i, j].tolist() + [self.eot]
                    sequences[tuple(sequence)] = sum_logprobs[i][j].item()
                    if len(sequences) >= self.beam_size:
                        break

        tokens: List[List[Tensor]] = [
            [torch.tensor(seq) for seq in sequences.keys()] for sequences in self.finished_sequences
        ]
        sum_logprobs: List[List[float]] = [
            list(sequences.values()) for sequences in self.finished_sequences
        ]
        return tokens, sum_logprobs


class BeamSearchDecoder(TokenDecoder):
    def __init__(
        self,
        beam_size: int,
        eot: int,
        inference: Inference,
        patience: Optional[float] = None,
    ):
        self.beam_size = beam_size
        self.eot = eot
        self.inference = inference
        self.patience = patience or 1.0
        self.max_candidates: int = round(beam_size * self.patience)
        self.finished_sequences = None
        with open("/home/marvin.rajwadi@CHASEITS/KenLM/juju_test/engchar.txt", 'r') as file:
            # Read each line from the file and append to the list
            data_list = file.readlines()
            print("datalist loaded")

        # Strip newline characters from each element
        self.data_list = [int(item.strip()) for item in data_list]
        self.tokenizer = get_tokenizer('en')
        assert (
            self.max_candidates > 0
        ), f"Invalid beam size ({beam_size}) or patience ({patience})"

    def reset(self):
        self.finished_sequences = None


    def update(
        self, tokens: Tensor, logits: Tensor, sum_logprobs: Tensor
    ) -> Tuple[Tensor, bool]:
        if tokens.shape[0] % self.beam_size != 0:
            raise ValueError(f"{tokens.shape}[0] % {self.beam_size} != 0")

        n_audio = tokens.shape[0] // self.beam_size
        if self.finished_sequences is None:  # for the first update
            self.finished_sequences = [{} for _ in range(n_audio)]

        logprobs = F.log_softmax(logits.float(), dim=-1)
        # with open("/home/marvin.rajwadi@CHASEITS/miniconda3/envs/kenlm/lib/python3.10/site-packages/whisper/whisper-KenLM/engchar.txt", 'r') as file:
        #     # Read each line from the file and append to the list
        #     data_list = file.readlines()

        # # Strip newline characters from each element
        # data_list = [int(item.strip()) for item in data_list]
        next_tokens, source_indices, finished_sequences = [], [], []
        for i in range(n_audio):
            scores, sources, finished = {}, {}, {}

            # STEP 1: calculate the cumulative log probabilities for possible candidates
            for j in range(self.beam_size):
                idx = i * self.beam_size + j
                prefix = tokens[idx].tolist()
                print(self.tokenizer.decode(prefix))
                # for idx_,c in enumerate(logprobs[idx]):
                #     if idx_ !=self.data_list:
                #         logprobs[idx][idx_]=float("-inf")
                for logprob, token in zip(*logprobs[idx].topk(self.beam_size + 1)):
                    print(f"whisper top k: {self.tokenizer.decode([token])} {logprob}")
                    new_logprob = (sum_logprobs[idx] + logprob).item()
                    sequence = tuple(prefix + [token.item()])
                    scores[sequence] = new_logprob
                    sources[sequence] = idx

            # STEP 2: rank the candidates and keep the top beam_size sequences for each audio
            saved = 0
            for sequence in sorted(scores, key=scores.get, reverse=True):
                if sequence[-1] == self.eot:
                    finished[sequence] = scores[sequence]
                else:
                    sum_logprobs[len(next_tokens)] = scores[sequence]
                    next_tokens.append(sequence)
                    source_indices.append(sources[sequence])

                    saved += 1
                    if saved == self.beam_size:
                        break

            finished_sequences.append(finished)

        tokens = torch.tensor(next_tokens, device=tokens.device)
        self.inference.rearrange_kv_cache(source_indices)

        # add newly finished sequences to self.finished_sequences
        assert len(self.finished_sequences) == len(finished_sequences)
        for previously_finished, newly_finished in zip(
            self.finished_sequences, finished_sequences
        ):
            for seq in sorted(newly_finished, key=newly_finished.get, reverse=True):
                if len(previously_finished) >= self.max_candidates:
                    break  # the candidate list is full
                previously_finished[seq] = newly_finished[seq]

        # mark as completed if all audio has enough number of samples
        completed = all(
            len(sequences) >= self.max_candidates
            for sequences in self.finished_sequences
        )
        return tokens, completed

    def finalize(self, preceding_tokens: Tensor, sum_logprobs: Tensor):
        # collect all finished sequences, including patience, and add unfinished ones if not enough
        sum_logprobs = sum_logprobs.cpu()
        for i, sequences in enumerate(self.finished_sequences):
            if (
                len(sequences) < self.beam_size
            ):  # when not enough sequences are finished
                for j in list(np.argsort(sum_logprobs[i]))[::-1]:
                    sequence = preceding_tokens[i, j].tolist() + [self.eot]
                    sequences[tuple(sequence)] = sum_logprobs[i][j].item()
                    if len(sequences) >= self.beam_size:
                        break

        tokens: List[List[Tensor]] = [
            [torch.tensor(seq) for seq in sequences.keys()]
            for sequences in self.finished_sequences
        ]
        sum_logprobs: List[List[float]] = [
            list(sequences.values()) for sequences in self.finished_sequences
        ]
        return tokens, sum_logprobs


class LogitFilter:
    def apply(self, logits: Tensor, tokens: Tensor) -> None:
        """Apply any filtering or masking to logits in-place

        Parameters
        ----------
        logits : Tensor, shape = (n_batch, vocab_size)
            per-token logits of the probability distribution at the current step

        tokens : Tensor, shape = (n_batch, current_sequence_length)
            all tokens in the context so far, including the prefix and sot_sequence tokens

        """
        raise NotImplementedError


class SuppressBlank(LogitFilter):
    def __init__(self, tokenizer: Tokenizer, sample_begin: int):
        self.tokenizer = tokenizer
        self.sample_begin = sample_begin

    def apply(self, logits: Tensor, tokens: Tensor):
        if tokens.shape[1] == self.sample_begin:
            logits[:, self.tokenizer.encode(" ") + [self.tokenizer.eot]] = -np.inf


class SuppressTokens(LogitFilter):
    def __init__(self, suppress_tokens: Sequence[int]):
        self.suppress_tokens = list(suppress_tokens)

    def apply(self, logits: Tensor, tokens: Tensor):
        logits[:, self.suppress_tokens] = -np.inf


class ApplyTimestampRules(LogitFilter):
    def __init__(
        self,
        tokenizer: Tokenizer,
        sample_begin: int,
        max_initial_timestamp_index: Optional[int],
    ):
        self.tokenizer = tokenizer
        self.sample_begin = sample_begin
        self.max_initial_timestamp_index = max_initial_timestamp_index

    def apply(self, logits: Tensor, tokens: Tensor):
        # suppress <|notimestamps|> which is handled by without_timestamps
        if self.tokenizer.no_timestamps is not None:
            logits[:, self.tokenizer.no_timestamps] = -np.inf

        # timestamps have to appear in pairs, except directly before EOT; mask logits accordingly
        for k in range(tokens.shape[0]):
            sampled_tokens = tokens[k, self.sample_begin :]
            seq = [t for t in sampled_tokens.tolist()]
            last_was_timestamp = (
                len(seq) >= 1 and seq[-1] >= self.tokenizer.timestamp_begin
            )
            penultimate_was_timestamp = (
                len(seq) < 2 or seq[-2] >= self.tokenizer.timestamp_begin
            )

            if last_was_timestamp:
                if penultimate_was_timestamp:  # has to be non-timestamp
                    logits[k, self.tokenizer.timestamp_begin :] = -np.inf
                else:  # cannot be normal text tokens
                    logits[k, : self.tokenizer.eot] = -np.inf

            timestamps = sampled_tokens[
                sampled_tokens.ge(self.tokenizer.timestamp_begin)
            ]
            if timestamps.numel() > 0:
                # timestamps shouldn't decrease; forbid timestamp tokens smaller than the last
                # also force each segment to have a nonzero length, to prevent infinite looping
                if last_was_timestamp and not penultimate_was_timestamp:
                    timestamp_last = timestamps[-1]
                else:
                    timestamp_last = timestamps[-1] + 1
                logits[k, self.tokenizer.timestamp_begin : timestamp_last] = -np.inf

        if tokens.shape[1] == self.sample_begin:
            # suppress generating non-timestamp tokens at the beginning
            logits[:, : self.tokenizer.timestamp_begin] = -np.inf

            # apply the `max_initial_timestamp` option
            if self.max_initial_timestamp_index is not None:
                last_allowed = (
                    self.tokenizer.timestamp_begin + self.max_initial_timestamp_index
                )
                logits[:, last_allowed + 1 :] = -np.inf

        # if sum of probability over timestamps is above any other token, sample timestamp
        logprobs = F.log_softmax(logits.float(), dim=-1)
        for k in range(tokens.shape[0]):
            timestamp_logprob = logprobs[k, self.tokenizer.timestamp_begin :].logsumexp(
                dim=-1
            )
            max_text_token_logprob = logprobs[k, : self.tokenizer.timestamp_begin].max()
            if timestamp_logprob > max_text_token_logprob:
                logits[k, : self.tokenizer.timestamp_begin] = -np.inf


class DecodingTask:
    inference: Inference
    sequence_ranker: SequenceRanker
    decoder: TokenDecoder
    logit_filters: List[LogitFilter]

    def __init__(self, model: "Whisper", options: DecodingOptions):
        self.model = model

        language = options.language or "en"
        tokenizer = get_tokenizer(
            model.is_multilingual,
            num_languages=model.num_languages,
            language=language,
            task=options.task,
        )
        self.tokenizer: Tokenizer = tokenizer
        self.options: DecodingOptions = self._verify_options(options)

        self.n_group: int = options.beam_size or options.best_of or 1
        self.n_ctx: int = model.dims.n_text_ctx
        self.sample_len: int = options.sample_len or model.dims.n_text_ctx // 2

        self.sot_sequence: Tuple[int] = tokenizer.sot_sequence
        if self.options.without_timestamps:
            self.sot_sequence = tokenizer.sot_sequence_including_notimestamps

        self.initial_tokens: Tuple[int] = self._get_initial_tokens()
        self.sample_begin: int = len(self.initial_tokens)
        self.sot_index: int = self.initial_tokens.index(tokenizer.sot)

        # inference: implements the forward pass through the decoder, including kv caching
        self.inference = PyTorchInference(model, len(self.initial_tokens))

        # sequence ranker: implements how to rank a group of sampled sequences
        self.sequence_ranker = MaximumLikelihoodRanker(options.length_penalty)

        # decoder: implements how to select the next tokens, given the autoregressive distribution
        if options.beam_size is not None:
            if (options.withlm):
                if options.select_candidates is not None:
                    print(f"Running KenLM on selected {options.select_candidates} candidates.")
                else:
                    print(f"Running KenLM on all {50365} tokens.")
                self.decoder = BeamSearchDecoderWithLM(
                    options.beam_size, tokenizer.eot, self.inference, options.patience, 
                    options.lm_path, options.lm_alpha, options.lm_beta, options.select_candidates
                )
            else:
                self.decoder = BeamSearchDecoder(
                    options.beam_size, tokenizer.eot, self.inference, options.patience
                )
        else:
            self.decoder = GreedyDecoder(options.temperature, tokenizer.eot)

        # logit filters: applies various rules to suppress or penalize certain tokens
        self.logit_filters = []
        if self.options.suppress_blank:
            self.logit_filters.append(SuppressBlank(self.tokenizer, self.sample_begin))
        if self.options.suppress_tokens:
            self.logit_filters.append(SuppressTokens(self._get_suppress_tokens()))
        if not options.without_timestamps:
            precision = CHUNK_LENGTH / model.dims.n_audio_ctx  # usually 0.02 seconds
            max_initial_timestamp_index = None
            if options.max_initial_timestamp:
                max_initial_timestamp_index = round(
                    self.options.max_initial_timestamp / precision
                )
            self.logit_filters.append(
                ApplyTimestampRules(
                    tokenizer, self.sample_begin, max_initial_timestamp_index
                )
            )

    def _verify_options(self, options: DecodingOptions) -> DecodingOptions:
        if options.beam_size is not None and options.best_of is not None:
            raise ValueError("beam_size and best_of can't be given together")
        if options.temperature == 0:
            if options.best_of is not None:
                raise ValueError("best_of with greedy sampling (T=0) is not compatible")
        if options.patience is not None and options.beam_size is None:
            raise ValueError("patience requires beam_size to be given")
        if options.length_penalty is not None and not (
            0 <= options.length_penalty <= 1
        ):
            raise ValueError("length_penalty (alpha) should be a value between 0 and 1")

        return options

    def _get_initial_tokens(self) -> Tuple[int]:
        tokens = list(self.sot_sequence)

        if prefix := self.options.prefix:
            prefix_tokens = (
                self.tokenizer.encode(" " + prefix.strip())
                if isinstance(prefix, str)
                else prefix
            )
            if self.sample_len is not None:
                max_prefix_len = self.n_ctx // 2 - self.sample_len
                prefix_tokens = prefix_tokens[-max_prefix_len:]
            tokens = tokens + prefix_tokens

        if prompt := self.options.prompt:
            prompt_tokens = (
                self.tokenizer.encode(" " + prompt.strip())
                if isinstance(prompt, str)
                else prompt
            )
            tokens = (
                [self.tokenizer.sot_prev]
                + prompt_tokens[-(self.n_ctx // 2 - 1) :]
                + tokens
            )

        return tuple(tokens)

    def _get_suppress_tokens(self) -> Tuple[int]:
        suppress_tokens = self.options.suppress_tokens

        if isinstance(suppress_tokens, str):
            suppress_tokens = [int(t) for t in suppress_tokens.split(",")]

        if -1 in suppress_tokens:
            suppress_tokens = [t for t in suppress_tokens if t >= 0]
            suppress_tokens.extend(self.tokenizer.non_speech_tokens)
        elif suppress_tokens is None or len(suppress_tokens) == 0:
            suppress_tokens = []  # interpret empty string as an empty list
        else:
            assert isinstance(suppress_tokens, list), "suppress_tokens must be a list"

        suppress_tokens.extend(
            [
                self.tokenizer.transcribe,
                self.tokenizer.translate,
                self.tokenizer.sot,
                self.tokenizer.sot_prev,
                self.tokenizer.sot_lm,
            ]
        )
        if self.tokenizer.no_speech is not None:
            # no-speech probability is collected separately
            suppress_tokens.append(self.tokenizer.no_speech)

        return tuple(sorted(set(suppress_tokens)))

    def _get_audio_features(self, mel: Tensor):
        if self.options.fp16:
            mel = mel.half()

        if mel.shape[-2:] == (
            self.model.dims.n_audio_ctx,
            self.model.dims.n_audio_state,
        ):
            # encoded audio features are given; skip audio encoding
            audio_features = mel
        else:
            audio_features = self.model.encoder(mel)

        if audio_features.dtype != (
            torch.float16 if self.options.fp16 else torch.float32
        ):
            return TypeError(
                f"audio_features has an incorrect dtype: {audio_features.dtype}"
            )

        return audio_features

    def _detect_language(self, audio_features: Tensor, tokens: Tensor):
        languages = [self.options.language] * audio_features.shape[0]
        lang_probs = None

        if self.options.language is None or self.options.task == "lang_id":
            lang_tokens, lang_probs = self.model.detect_language(
                audio_features, self.tokenizer
            )
            languages = [max(probs, key=probs.get) for probs in lang_probs]
            if self.options.language is None:
                tokens[:, self.sot_index + 1] = lang_tokens  # write language tokens

        return languages, lang_probs

    def _main_loop(self, audio_features: Tensor, tokens: Tensor):
        n_batch = tokens.shape[0]
        sum_logprobs: Tensor = torch.zeros(n_batch, device=audio_features.device)
        no_speech_probs = [np.nan] * n_batch

        try:
            for i in range(self.sample_len):
                logits = self.inference.logits(tokens, audio_features)

                if (
                    i == 0 and self.tokenizer.no_speech is not None
                ):  # save no_speech_probs
                    probs_at_sot = logits[:, self.sot_index].float().softmax(dim=-1)
                    no_speech_probs = probs_at_sot[:, self.tokenizer.no_speech].tolist()

                # now we need to consider the logits at the last token only
                logits = logits[:, -1]

                # apply the logit filters, e.g. for suppressing or applying penalty to
                for logit_filter in self.logit_filters:
                    logit_filter.apply(logits, tokens)

                # expand the tokens tensor with the selected next tokens
                tokens, completed = self.decoder.update(tokens, logits, sum_logprobs)

                if completed or tokens.shape[-1] > self.n_ctx:
                    break
        finally:
            self.inference.cleanup_caching()

        return tokens, sum_logprobs, no_speech_probs

    @torch.no_grad()
    def run(self, mel: Tensor) -> List[DecodingResult]:
        self.decoder.reset()
        tokenizer: Tokenizer = self.tokenizer
        n_audio: int = mel.shape[0]

        audio_features: Tensor = self._get_audio_features(mel)  # encoder forward pass
        tokens: Tensor = torch.tensor([self.initial_tokens]).repeat(n_audio, 1)

        # detect language if requested, overwriting the language token
        languages, language_probs = self._detect_language(audio_features, tokens)
        if self.options.task == "lang_id":
            return [
                DecodingResult(
                    audio_features=features, language=language, language_probs=probs
                )
                for features, language, probs in zip(
                    audio_features, languages, language_probs
                )
            ]

        # repeat text tensors by the group size, for beam search or best-of-n sampling
        tokens = tokens.repeat_interleave(self.n_group, dim=0).to(audio_features.device)

        # call the main sampling loop
        tokens, sum_logprobs, no_speech_probs = self._main_loop(audio_features, tokens)

        # reshape the tensors to have (n_audio, n_group) as the first two dimensions
        audio_features = audio_features[:: self.n_group]
        no_speech_probs = no_speech_probs[:: self.n_group]
        assert audio_features.shape[0] == len(no_speech_probs) == n_audio

        tokens = tokens.reshape(n_audio, self.n_group, -1)
        sum_logprobs = sum_logprobs.reshape(n_audio, self.n_group)

        # get the final candidates for each group, and slice between the first sampled token and EOT
        tokens, sum_logprobs = self.decoder.finalize(tokens, sum_logprobs)
        tokens: List[List[Tensor]] = [
            [t[self.sample_begin : (t == tokenizer.eot).nonzero()[0, 0]] for t in s]
            for s in tokens
        ]
        tokens_en=[]
        tokens_en_sc=[]
        tokens_bs=[]
        tokens_bs_sc =[]
        import fasttext
        from langdetect import detect


        class LanguageIdentification:
        
            def __init__(self):
                pretrained_lang_model = "/home/marvin.rajwadi@CHASEITS/KenLM/juju_test/fasttext/lid.176.ftz"
                self.model = fasttext.load_model(pretrained_lang_model)
        
            def predict_lang(self, text):
                predictions = self.model.predict(text, k=1) # returns top 2 matching languages
                return predictions
        LANGUAGE = LanguageIdentification()
        for idx,te in enumerate([self.tokenizer.decode(j) for j in tokens[0]]):
            
            #lang = LANGUAGE.predict_lang(te)
            #pred_lang=lang[0][0].split("__")[-1].strip()
            try:
                pred_lang=detect(te)
                print(pred_lang)
            except:
                pred_lang='unk'
            if pred_lang != "en":# or "danish" in te.lower() or "english" in te.lower() or "translate" in te.lower() or "translation" in te.lower():
                sum_logprobs[0][idx] = sum_logprobs[0][idx] -7.0
                #tokens_en.append(tokens[0][idx])
                #tokens_en_sc.append(sum_logprobs[0][idx])
            # else:
            #     #tokens_bs.append(tokens[0][idx])
            #     #tokens_bs_sc.append(sum_logprobs[0][idx])
            #     length_factor = 1 + 0.1 * np.sum([1.0 / len(te.split())])
    
            #     # Apply the length boost factor to the sum of log probabilities
            #    # boosted_sum = sum_log_probs * length_factor
            #     sum_logprobs[0][idx] = sum_logprobs[0][idx] / length_factor

        
        if True:
            print("Top 5 segments: ",[self.tokenizer.decode(j) for j in tokens[0]])
            print("top 5 prob: ", sum_logprobs)
        # select the top-ranked sample in each group
        selected = self.sequence_ranker.rank(tokens, sum_logprobs)
        #selected_en = self.sequence_ranker.rank(tokens_en, tokens_en_sc)
        #selected_bs = self.sequence_ranker.rank(tokens_bs, tokens_bs_sc)
        tokens: List[List[int]] = [t[i].tolist() for i, t in zip(selected, tokens)]
        texts: List[str] = [tokenizer.decode(t).strip() for t in tokens]
        #if self.options.debug:
        if True:
            print("selected: ", selected)
            print("top ranked segment: ", texts)

        sum_logprobs: List[float] = [lp[i] for i, lp in zip(selected, sum_logprobs)]
        avg_logprobs: List[float] = [
            lp / (len(t) + 1) for t, lp in zip(tokens, sum_logprobs)
        ]

        fields = (
            texts,
            languages,
            tokens,
            audio_features,
            avg_logprobs,
            no_speech_probs,
        )
        if len(set(map(len, fields))) != 1:
            raise RuntimeError(f"inconsistent result lengths: {list(map(len, fields))}")

        return [
            DecodingResult(
                audio_features=features,
                language=language,
                tokens=tokens,
                text=text,
                avg_logprob=avg_logprob,
                no_speech_prob=no_speech_prob,
                temperature=self.options.temperature,
                compression_ratio=compression_ratio(text),
            )
            for text, language, tokens, features, avg_logprob, no_speech_prob in zip(
                *fields
            )
        ]


@torch.no_grad()
def decode(
    model: "Whisper",
    mel: Tensor,
    options: DecodingOptions = DecodingOptions(),
    **kwargs,
) -> Union[DecodingResult, List[DecodingResult]]:
    """
    Performs decoding of 30-second audio segment(s), provided as Mel spectrogram(s).

    Parameters
    ----------
    model: Whisper
        the Whisper model instance

    mel: torch.Tensor, shape = (80, 3000) or (*, 80, 3000)
        A tensor containing the Mel spectrogram(s)

    options: DecodingOptions
        A dataclass that contains all necessary options for decoding 30-second segments

    Returns
    -------
    result: Union[DecodingResult, List[DecodingResult]]
        The result(s) of decoding contained in `DecodingResult` dataclass instance(s)
    """
    if single := mel.ndim == 2:
        mel = mel.unsqueeze(0)

    if kwargs:
        options = replace(options, **kwargs)
    print(f"temperature: {options.temperature}")
    result = DecodingTask(model, options).run(mel)

    return result[0] if single else result