import pandas as pd
import time
import warnings
import numpy as np
import distance
import transition_system
import argparse
from itertools import product

warnings.filterwarnings("ignore")


def predict_trace_beam(prefix, data_train, beam_width):
    max_sequence_length = data_train.length.max()
    suffix_trace = {}
    suffix_time = {}
    inp = {}
    pred_act = {}
    for k in range(1, beam_width + 1):
        suffix_trace['beam {}'.format(k)] = {}
        suffix_trace['beam {}'.format(k)]['pred'] = []
        suffix_trace['beam {}'.format(k)]['prob'] = []
        inp['beam {}'.format(k)] = prefix

    for pos in range(max_sequence_length):
        if pos == 0:
            act = ts.predict_next(inp['beam {}'.format(k)], best_only=False)
            for k in range(1, beam_width + 1):
                key_val = list(dict(sorted(act.items(), key=lambda item: item[1])).items())[-k]

                suffix_trace['beam {}'.format(k)]['pred'].append(key_val[0])
                suffix_trace['beam {}'.format(k)]['prob'].append(key_val[1])
                inp['beam {}'.format(k)] = np.append(inp['beam {}'.format(k)],
                                                     suffix_trace['beam {}'.format(k)]['pred'][-1])

        else:
            if ts.EOS in suffix_trace['beam 1']['pred']:
                break
            for k in range(1, beam_width + 1):
                pred_act['beam {}'.format(k)] = ts.predict_next(inp['beam {}'.format(k)],
                                                                best_only=False)
            for k in range(1, beam_width + 1):
                if k == 1:
                    stack = [(np.sum(-np.log(suffix_trace['beam {}'.format(k)]['prob'])) - np.log(
                        pred_act['beam {}'.format(k)][key])) / (pos ** 0.6) for key in
                             pred_act['beam {}'.format(k)].keys()]
                    stack = np.array(stack)
                else:
                    stack = np.hstack((stack, [(np.sum(
                        -np.log(suffix_trace['beam {}'.format(k)]['prob'])) - np.log(
                        pred_act['beam {}'.format(k)][key])) / (pos ** 0.6) for key in
                                               pred_act['beam {}'.format(k)].keys()]))

            indicies = stack.argsort()[:beam_width].tolist()
            temp_pred = {}
            temp_prob = {}
            for k in range(beam_width):
                best_k = int(np.floor(indicies[k] / ts.vocab_size) + 1)
                pos_k = indicies[k] - ts.vocab_size * int(np.floor(indicies[k] / ts.vocab_size))
                temp_pred['beam {}'.format(k + 1)] = suffix_trace['beam {}'.format(best_k)][
                                                         'pred'] + [list(
                    pred_act['beam {}'.format(best_k)].keys())[pos_k]]
                temp_prob['beam {}'.format(k + 1)] = suffix_trace['beam {}'.format(best_k)][
                                                         'prob'] + [list(
                    pred_act['beam {}'.format(best_k)].values())[pos_k]]
            for k in range(1, beam_width + 1):
                suffix_trace['beam {}'.format(k)]['pred'] = temp_pred['beam {}'.format(k)]
                suffix_trace['beam {}'.format(k)]['prob'] = temp_prob['beam {}'.format(k)]

            for k in range(1, beam_width + 1):
                inp['beam {}'.format(k)] = np.append(inp['beam {}'.format(k)],
                                                     suffix_trace['beam {}'.format(k)]['pred'][-1])
    return suffix_trace, suffix_time


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    print("Read event logs.")
    parser.add_argument('--training_log', required=True, help='Path to training log CSV')
    parser.add_argument('--test_log', required=True, help='Path to test log CSV')
    args = parser.parse_args()
    print("Start constructing transition system-based predictor.")
    data_train = pd.read_csv(args.training_log, index_col='CaseID')
    data_test = pd.read_csv(args.test_log, index_col='CaseID')
    vocabulary = list(set([act for trace in data_train.trace for act in trace.split(', ')]))
    vocab_size = len(vocabulary)

    config = {'window': 2, 'beam': 3}
    ts = transition_system.TransitionSystemPredictor(data_train, vocab_size=vocab_size,
                                                     window=config['window'])
    trace_actual = []
    trace_predicted = []
    time_actual = []
    time_predicted = []
    trace_in = []
    ids = []
    t = time.time()

    for trace in data_test.iterrows():
        trace_acts = trace[1]['trace'].split(', ')
        for pos in range(1, len(trace_acts)):
            prefix = trace_acts[:pos]
            suf_trace, suf_time = predict_trace_beam(prefix, data_train,
                                                     beam_width=config['beam'])

            trace_predicted.append(suf_trace['beam 1']['pred'])
            trace_actual.append(trace_acts[pos:])
            trace_in.append(prefix)

            ids.append(trace[0])

    for cut in range(len(trace_predicted)):
        if '<EOS>' in trace_predicted[cut]:
            trace_predicted[cut] = trace_predicted[cut][
                                   :trace_predicted[cut].index('<EOS>') + 1]

    d_result = pd.DataFrame(ids)
    d_result = d_result.rename(columns={0: "CaseID"})
    d_result['trace_prediction'] = trace_predicted
    d_result['trace_actual'] = trace_actual
    d_result['prefix'] = trace_in
    d_result['counter'] = d_result.groupby(
        (d_result['CaseID'] != d_result['CaseID'].shift(1)).cumsum()).cumcount()

    for col in ['trace_prediction', 'trace_actual']:
        d_result[col] = d_result[col].apply(lambda x: ''.join(str(x)))
        d_result[col] = d_result[col].apply(
            lambda x: x.lstrip('\[').rstrip('\]').replace('\'', ''))

    d_result = d_result[(d_result['trace_actual'] != '')]
    d_result = d_result[(d_result['trace_actual'] != '<EOS>')]
    d_result['levenshtein'] = d_result.apply(
        lambda x: 1 - distance.nlevenshtein(x['trace_prediction'].split(', '),
                                            x['trace_actual'].split(', ')), axis=1)

    ts = transition_system.TransitionSystemPredictor(data_train, vocab_size=vocab_size,
                                                     window=config['window'])
    trace_actual = []
    trace_predicted = []
    time_actual = []
    time_predicted = []
    trace_in = []
    ids = []
    t = time.time()

    for trace in data_test.iterrows():

        trace_acts = trace[1]['trace'].split(', ')
        for pos in range(1, len(trace_acts)):
            prefix = trace_acts[:pos]
            suf_trace, suf_time = predict_trace_beam(prefix, data_train, beam_width=config['beam'])

            trace_predicted.append(suf_trace['beam 1']['pred'])
            trace_actual.append(trace_acts[pos:])
            trace_in.append(prefix)
            ids.append(trace[0])
    elapsed = time.time() - t
    for cut in range(len(trace_predicted)):
        if '<EOS>' in trace_predicted[cut]:
            trace_predicted[cut] = trace_predicted[cut][
                                   :trace_predicted[cut].index('<EOS>') + 1]
            # time_predicted[cut] = time_predicted[cut][:len(trace_predicted[cut])]

    d_result = pd.DataFrame(ids)
    d_result = d_result.rename(columns={0: "CaseID"})
    d_result['trace_prediction'] = trace_predicted
    d_result['trace_actual'] = trace_actual
    d_result['prefix'] = trace_in
    for col in ['trace_prediction', 'trace_actual']:
        d_result[col] = d_result[col].apply(lambda x: ''.join(str(x)))
        d_result[col] = d_result[col].apply(
            lambda x: x.lstrip('\[').rstrip('\]').replace('\'', ''))

    d_result = d_result[(d_result['trace_actual'] != '')]
    d_result = d_result[(d_result['trace_actual'] != '<EOS>')]
    d_result['levenshtein'] = d_result.apply(
        lambda x: 1 - distance.nlevenshtein(x['trace_prediction'].split(', '),
                                            x['trace_actual'].split(', ')), axis=1)
    print('Damerau–Levenshtein similarity (DLS) for test log:', np.mean(d_result.groupby('CaseID')['levenshtein'].mean()))