import math
import torch.nn.functional as F
import collections
import torch
import numpy
from collections import defaultdict
from dataclasses import dataclass
from fairseq import metrics
# 
# from fairseq.sequence_generator  import SequenceGenerator
# from sequence_generator import 
from fairseq import utils #, bleu
from fairseq.dataclass import FairseqDataclass
from fairseq.criterions import FairseqCriterion, register_criterion
from omegaconf import II
from fairseq.models import FairseqMultiModel
from fairseq.sequence_generator import SequenceGenerator


@register_criterion('v2')
class V2Criterion(FairseqCriterion):

    def __init__(self, task, sentence_avg): #(self, args, task):
        super().__init__(task)
        self.cnt=0

    # def forward(self, model, sample, reduce=True):
    #     """Compute the loss for the given sample.
    #
    #     Returns a tuple with three elements:
    #     1) the loss
    #     2) the sample size, which is used as the denominator for the gradient
    #     3) logging outputs to display while training
    #     """
    #     net_output = model(**sample['net_input'])
    #     lprobs = model.get_normalized_probs(net_output, log_probs=True)
    #     lprobs = lprobs.view(-1, lprobs.size(-1))
    #     target = model.get_targets(sample, net_output).view(-1)
    #     loss = F.nll_loss(lprobs, target, size_average=False,
    #                       ignore_index=self.padding_idx,
    #                       reduce=reduce)
    #     sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
    #     logging_output = {
    #         'loss': utils.item(loss.data) if reduce else loss.data,
    #         'ntokens': sample['ntokens'],
    #         'sample_size': sample_size,
    #     }
    #     return loss, sample_size, logging_output

    def forward(self, model, sample, reduce=True):
        # sample mode
        #print('!!!RL loss.')
        model.eval()
        # src_dict = self.task.source_dictionary
        tgt_dict = self.task.target_dictionary
        eos_idx = self.task.target_dictionary.eos()
        sample_beam = 4 #self.args.sample_beam

        translator = SequenceGenerator([model], tgt_dict=tgt_dict,beam_size=sample_beam)

        translator.cuda()
        ct = 0
        translations = []

        s = utils.move_to_cuda(sample)
        input = s['net_input']
        max_len = 200
        with torch.no_grad():
            hypos = translator.generate([model],s)
        for i, id in enumerate(s['id'].data):
            src = input['src_tokens'].data[i, :]
            # remove padding from ref
            ref = utils.strip_pad(s['target'].data[i, :], tgt_dict.pad()) if s['target'] is not None else None
            translations.append((id, src, ref, hypos[i]))

            ct += 1
        # print("sample batch size:", ct)

        model.train()

        # MLE loss
        mle_net_output = model(**sample['net_input'])
        mle_lprobs = model.get_normalized_probs(mle_net_output, log_probs=True)
        mle_lprobs = mle_lprobs.view(-1, mle_lprobs.size(-1))
        mle_target = model.get_targets(sample, mle_net_output).view(-1)
        mle_loss = F.nll_loss(mle_lprobs, mle_target, size_average=False,
                              ignore_index=self.padding_idx, reduce=reduce)
        mle_tokens = sample['ntokens']
        avg_mle_loss = mle_loss / mle_tokens
        self.cnt  += 1
        if self.cnt % 4 !=0 : 
            total_loss = avg_mle_loss
            total_tokens =  sample["ntokens"]
            logging_output = {
                'loss': utils.item(total_loss.data),
                'ntokens': total_tokens,
                'sample_size': total_tokens,
            }
    #         print('total: ',total_loss)
            return total_loss, total_tokens, logging_output

    
        print('avg_mle_loss:', avg_mle_loss)
        # RL loss
        batch_rl_loss = 0
        batch_tokens = 0
        sample_ind = 0
        for sample_id, src_tokens, tgt_tokens, hypos in translations:
            # calculate bleu
            sample_ind += 1
            rewards = torch.Tensor(sample_beam).float().cuda()
            logprobs = torch.Tensor(sample_beam).float().cuda()
            for i in range(sample_beam):
                hypo = hypos[i]
                trans_tokens = hypo['tokens']
                rewards[i] = self.compute_gleu(tgt_tokens.cpu(), trans_tokens.cpu(), max_order=4, gram=0).cuda()
                # one_sample loss calculation
                tgt_input_tokens = trans_tokens.new(trans_tokens.shape).fill_(0)
                assert trans_tokens[-1] == eos_idx
                tgt_input_tokens[0] = eos_idx
                tgt_input_tokens[1:] = trans_tokens[:-1]
                train_sample = {
                    'net_input': {
                        'src_tokens': src_tokens.view(1, -1),
                        'src_lengths': torch.LongTensor(src_tokens.numel()).view(1, -1),
                        'prev_output_tokens': tgt_input_tokens.view(1, -1),
                    },
                    'target': trans_tokens.view(1, -1)
                }
                train_sample = utils.move_to_cuda(train_sample)
                net_output = model(**train_sample['net_input'])
                lprobs = model.get_normalized_probs(net_output, log_probs=True)
                lprobs = lprobs.view(-1, lprobs.size(-1))
                target = model.get_targets(train_sample, net_output).view(-1, 1)
                non_pad_mask = target.ne(tgt_dict.pad())
                lprob = -lprobs.gather(dim=-1, index=target)[non_pad_mask]
                logprobs[i] = torch.sum(lprob)
                ntokens = len(train_sample['target'])
                batch_tokens += ntokens
            rl_loss = torch.sum(logprobs * (rewards - rewards.mean()))  # one sample loss            
            batch_rl_loss += rl_loss
        
        avg_rl_loss = batch_rl_loss / batch_tokens
        print('avg_rl_loss:', avg_rl_loss)
        if True:
            total_loss = avg_mle_loss +  avg_rl_loss
            total_tokens = batch_tokens + mle_tokens
        else:
            total_loss = avg_rl_loss
            total_tokens = batch_tokens
        logging_output = {
            'loss': utils.item(total_loss.data),
            'ntokens': total_tokens,
            'sample_size': total_tokens,
        }
        print('total: ',total_loss)
        return total_loss, total_tokens, logging_output


    def _get_ngrams(self, segment, max_order):
        ngram_counts = collections.Counter()
        for order in range(1, max_order + 1):
            for i in range(0, len(segment) - order + 1):
                ngram = tuple(segment[i:i + order])
                ngram_counts[ngram] += 1
        return ngram_counts

    def compute_gleu(self, reference_corpus, translation_corpus, max_order=4, gram=0, smooth=False):
        scores = torch.zeros(max_order)
        reference_array = numpy.array(reference_corpus)
        translation_array = numpy.array(translation_corpus)
        matches_by_order = [0] * max_order
        possible_matches_by_order_ref = [0] * max_order
        possible_matches_by_order_trans = [0] * max_order
        reference_length = 0
        translation_length = 0
        reference_length += reference_array.shape[0]
        translation_length += translation_array.shape[0]
        merged_ref_ngram_counts = collections.Counter()
        merged_ref_ngram_counts |= self._get_ngrams(reference_array, max_order)
        translation_ngram_counts = self._get_ngrams(translation_array, max_order)
        overlap = translation_ngram_counts & merged_ref_ngram_counts
        for ngram in overlap:
            matches_by_order[len(ngram)-1] += overlap[ngram]
        for order in range(1, max_order+1):
            possible_matches_trans = translation_length - order + 1
            if possible_matches_trans > 0:
                possible_matches_by_order_trans[order-1] += possible_matches_trans
            possible_matches_ref = reference_length - order + 1
            if possible_matches_ref > 0:
                possible_matches_by_order_ref[order-1] += possible_matches_ref
        precisions = [0] * max_order
        recalls = [0] * max_order

        for i in range(0, max_order):
            if smooth:
                precisions[i] = ((matches_by_order[i] + 1.) / (possible_matches_by_order_trans[i] + 1.))
                recalls[i] = ((matches_by_order[i] + 1.) / (possible_matches_by_order_ref[i] + 1.))
            else:
                if possible_matches_by_order_trans[i] > 0:
                    precisions[i] = (float(matches_by_order[i]) / possible_matches_by_order_trans[i])
                else:
                    precisions[i] = 0.0
                
                if possible_matches_by_order_ref[i] > 0:
                    recalls[i] = (float(matches_by_order[i]) / possible_matches_by_order_ref[i])
                else:
                    recalls[i] = 0.0
        for i in range(max_order):
            scores[i] = min(precisions[i],recalls[i])

        if False :#self.args.modgleu:
            if reference_length < max_order and translation_length < max_order:
                order = max(reference_length, translation_length)
                scores = scores[0:order]
            else:
                order = max_order
        else:
            order = max_order
        if gram == 0:
            if min(scores) > 0:
                log_scores = torch.log(scores)
                p_log_sum = torch.sum((1. / order) * log_scores)
                geo_mean = torch.exp(p_log_sum)
                return geo_mean
            else:
                return torch.tensor(0.0)
        else:
            if scores[gram] > 0:
                return scores[gram]
            else:
                return torch.tensor(0.0)
            
    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        agg_output = {
            'loss': loss_sum / sample_size / math.log(2),
            'sample_size': sample_size,
        }
        if sample_size != ntokens:
            agg_output['nll_loss'] = loss_sum / ntokens / math.log(2)
        return agg_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        # we divide by log(2) to convert the loss from base e to base 2
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        if sample_size != ntokens:
            metrics.log_scalar(
                "nll_loss", loss_sum / ntokens / math.log(2), ntokens, round=3
            )
            metrics.log_derived(
                "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
            )
        else:
            metrics.log_derived(
                "ppl", lambda meters: utils.get_perplexity(meters["loss"].avg)
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
