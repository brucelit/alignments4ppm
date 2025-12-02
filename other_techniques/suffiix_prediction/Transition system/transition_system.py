class HashableDict(dict):
    def __init__(self, *args, **kwargs):
        super(HashableDict, self).__init__(*args, **kwargs)

    def __hash__(self):
        return hash(tuple(sorted(self.items())))


class TransitionSystemPredictor:
    EOS = '<EOS>'

    def __init__(self, log, vocab_size=28, window=1, statemaker='list', matcher=None):
        self.log = log
        self.window = window
        self.vocab_size = vocab_size
        if statemaker == 'set':
            self.statemaker = self._statemaker_set
        elif statemaker == 'list':
            self.statemaker = self._statemaker_list
        elif statemaker == 'bag':
            self.statemaker = self._statemaker_bag
        self.matcher = matcher if matcher is not None else self._matcher
        self.train()

    def _matcher(self, event):
        return True

    def _statemaker_set(self, prefix):
        return frozenset(prefix[-self.window:])

    def _statemaker_bag(self, prefix):
        return HashableDict(Counter(prefix[-self.window:]))

    def _statemaker_list(self, prefix):
        return tuple(prefix[-self.window:])

    def _trace_to_filtered_acts(self, trace):
        trace_activities = trace[1]['trace'].split(', ')
        return trace_activities

    def train(self):
        self.alfa = set([self.EOS])
        self.ts = {}
        self.ts_summary = {}
        for trace in self.log.iterrows():
            trace_activities = self._trace_to_filtered_acts(trace)
            self.alfa |= set(trace_activities)
            for pos in range(len(trace_activities)):
                state_here = self.statemaker(trace_activities[:pos])
                if not state_here in self.ts: self.ts[state_here] = []
                future = {
                    'next_event': trace_activities[pos],
                }
                self.ts[state_here].append(future)
        for state, futures in self.ts.items():
            next = {
                a: len([ns for ns in futures if ns['next_event'] == a]) / len(futures)
                for a in self.alfa
            }
            self.ts_summary[state] = {'next': next, 'support': len(futures)}

    def predict_next(self, prefix_activities, best_only=False):
        state_here = self.statemaker(prefix_activities)
        if state_here in self.ts_summary:
            result = self.ts_summary[state_here]['next']
        else:
            result = {a: 0 if a != self.EOS else 1.0 for a in self.alfa}
        if best_only:
            return sorted(result.items(), key=lambda kv: kv[1], reverse=True)[0]
        return result

    def predict_remaining(self, prefix):
        state_here = self.statemaker(prefix)

        if state_here in self.ts_summary:
            return self.ts_summary[state_here]['remaining']
        else:
            return float(0.0)

    def predict_support(self, prefix):
        prefix_activities = self._trace_to_filtered_acts(prefix)[:-1]
        state_here = self.statemaker(prefix_activities)
        if state_here in self.ts_summary:
            return self.ts_summary[state_here]['support']
        else:
            return 0.0


