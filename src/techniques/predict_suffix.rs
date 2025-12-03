use crate::{
    ebi_framework::displayable::Displayable, ebi_traits::{ebi_trait_finite_stochastic_language::EbiTraitFiniteStochasticLanguage, ebi_trait_stochastic_semantics::EbiTraitStochasticSemantics}, semantics::semantics::Semantics, stochastic_semantics::stochastic_semantics::StochasticSemantics, techniques::{astar_for_prediction, predict_next::get_model_trace},
};
use anyhow::{Result, anyhow};
use ebi_arithmetic::{MaybeExact, Zero, fraction::{fraction::Fraction}};
use ebi_objects::{Activity,ebi_objects::labelled_petri_net::TransitionIndex, ebi_objects::language_of_alignments::Move, ActivityKeyTranslator};
use std::{collections::HashMap, ops::{Add, AddAssign}};
use std::{
    fmt::{Debug, Display},
    hash::Hash,
};
use std::cmp::{max, min};

#[derive(Clone, Debug)]
pub struct StochasticWeightedCost {
    cost: f64,
    probability: f64,
    balance: f64,
}

impl Zero for StochasticWeightedCost {
    fn zero() -> Self {
        StochasticWeightedCost {
            cost: 0.0,
            probability: 1.0,
            balance: 0.0,
        }
    }

    fn is_zero(&self) -> bool {
        // self.stochastic_weighted_cost == 0.0
        self.cost == 0.0 && self.probability == 1.0
    }
}

impl Add for StochasticWeightedCost {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        let probability = &self.probability * &other.probability;
        let cost:f64 = self.cost + other.cost;
        StochasticWeightedCost {
            cost: cost,
            probability: probability.clone(),
            balance: 0.0,
        }
    }
}

impl AddAssign for StochasticWeightedCost {
    fn add_assign(&mut self, other: Self) {
        self.cost += other.cost;
        self.probability *= other.probability;
    }
}

impl Ord for StochasticWeightedCost {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // let self_stochastic_cost = self.cost ;
        // let other_stochastic_cost = other.cost;
        let self_stochastic_cost = (((1.0 +self.cost).ln()).powf(self.balance)) * ((1.0 - self.probability.ln()).powf(1.0-self.balance));
        let other_stochastic_cost = (((1.0 + other.cost).ln()).powf(other.balance)) * ((1.0 - other.probability.ln()).powf(1.0-other.balance));
        // let self_stochastic_cost = (((1.0 +self.cost).ln())) * ((1.0 - self.probability.ln()));
        // let other_stochastic_cost = (((1.0 + other.cost).ln())) * ((1.0 - other.probability.ln()));
        self_stochastic_cost
            .partial_cmp(&other_stochastic_cost)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

impl PartialOrd for StochasticWeightedCost {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for StochasticWeightedCost {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == std::cmp::Ordering::Equal
    }
}

impl Eq for StochasticWeightedCost {}

pub trait PredictSuffix {
    fn predict_suffix(
        &self,
        event_log: &Box<dyn EbiTraitFiniteStochasticLanguage>,
    ) -> Result<f64>;
}

impl PredictSuffix for EbiTraitStochasticSemantics {
    fn predict_suffix(
        &self,
        event_log: &Box<dyn EbiTraitFiniteStochasticLanguage>,
    ) -> Result<f64> {
        match self {
            EbiTraitStochasticSemantics::Usize(sem) => sem.predict_suffix(event_log),
            EbiTraitStochasticSemantics::Marking(sem) => sem.predict_suffix(event_log),
            EbiTraitStochasticSemantics::NodeStates(sem) => sem.predict_suffix(event_log),
        }
    }
}   

impl<State: Displayable> dyn StochasticSemantics<StoSemState = State, SemState = State, AliState = State> {
    pub fn predict_suffix(
        &self,
        event_log: &Box<dyn EbiTraitFiniteStochasticLanguage>,
    ) -> Result<f64> {
        let mut activity_key1 = self.activity_key().clone();
        let translator =
            ActivityKeyTranslator::new(&event_log.activity_key(), &mut activity_key1);
        

        let mut prefix_dls: HashMap<usize, f64> = HashMap::new();
        let mut prefix_prob: HashMap<usize, Fraction> = HashMap::new();
        let mut total_dls =0.0;
        let balance = 0.71; // balance between cost and probability

        for (trace, probability) in event_log.iter_trace_probability() {
            let trace2 = &translator.translate_trace(trace);

            // get the prefix and the next activity
            let mut prefix_collection = Vec::new();
            for i in 1..= trace2.len() {
                prefix_collection.push((trace2[..i].to_vec(), trace2[i..trace2.len()].to_vec()));
            }
            
            let mut temp_dls = 0.0;
            
            // iterate the prefix of the trace and align it
            for (prefix, suffix) in prefix_collection {
                println!("Aligning prefix: {:?} with suffix: {:?}", prefix, suffix);
                // get the start state
                if let Some(initial_state) = self.get_initial_state() {
                    let start = (0, initial_state);

                    // successor relation in the model
                    let successors = |(trace_index, state): &(usize, State)| {
                        let mut result: Vec<((usize, State), StochasticWeightedCost)> = vec![];

                        // log::debug!("successors of log {} model {}", trace_index, state);
                        if trace_index < &prefix.len() {
                            //we can do a log move
                            // log::debug!("\tlog move {}", trace[*trace_index]);

                            result.push((
                                (trace_index + 1, state.clone()),
                                StochasticWeightedCost {
                                    cost: 1.0,
                                    probability: 1.0,
                                    balance: balance,
                                    // stochastic_weighted_cost: 0.0
                                },
                            ));
                        }

                        //walk through the enabled transitions in the model
                        for transition in self.get_enabled_transitions(&state) {
                            let total_weight = self
                                .get_total_weight_of_enabled_transitions(&state)
                                .unwrap();

                            let mut new_state = state.clone();
                            // log::debug!("\t\tnew state before {}", new_state);
                            let _ = self.execute_transition(&mut new_state, transition);
                            // log::debug!("\t\tnew state after {}", new_state);

                            let transition_weight = self.get_transition_weight(&state, transition);
                            let transition_probability: f64 = (transition_weight / &total_weight).approx().unwrap();

                            if let Some(activity) = self.get_transition_activity(transition) {
                                //non-silent model move
                                result.push((
                                    (*trace_index, new_state.clone()),
                                    StochasticWeightedCost {
                                        cost: 1.0,
                                        probability: transition_probability,
                                        balance: balance,
                                        // stochastic_weighted_cost: 0.0
                                    },
                                ));
                                // log::debug!("\tmodel move t{} {} to {}", transition, activity, new_state);

                                //which may also be a synchronous move
                                if trace_index < &prefix.len() && activity == prefix[*trace_index] {
                                    //synchronous move
                                    // log::debug!("\tsynchronous move t{} {} to {}", transition, activity, new_state);
                                    result.push((
                                        (trace_index + 1, new_state),
                                        StochasticWeightedCost {
                                            cost: 0.0,
                                            probability: transition_probability,
                                            balance: balance,
                                            // stochastic_weighted_cost: 0.0
                                        },
                                    ));
                                }
                            } else {
                                //silent move
                                result.push((
                                    (*trace_index, new_state),
                                    StochasticWeightedCost {
                                        cost: 0.0,
                                        probability: transition_probability,
                                        balance: balance,
                                        // stochastic_weighted_cost: 0.0
                                    },
                                ));
                            }
                        }

                        // log::debug!("successors of {} {}: {:?}", trace_index, state, result);
                        result
                    };

                    //function that returns a heuristic on how far we are still minimally from a final state
                    let heuristic = |_astate: &(usize, State)| StochasticWeightedCost::zero();

                    //function that returns whether we are in a final synchronous product state
                    let success = |(trace_index, state): &(usize, State)| {
                        trace_index == &prefix.len() && self.is_final_state(&state)
                    };
                    match astar_for_prediction::astar(&start, successors, heuristic, success) {
                        Some((path, _cost)) => {
                            let mut predicted_suffix =  find_suffix_afterwards(self, path.clone());
                            if prefix.len() == 1 && predicted_suffix.len() ==0 {
                                predicted_suffix = get_model_trace(self, path.clone());
                            }
                            let full_model_trace = get_model_trace(self, path.clone());
                        // convert predicted_suffix to string
                        //     let predicted_suffix_str = predicted_suffix.iter().map(|activity| activity.to_string()).collect::<Vec<_>>().join(", ");
                        //    // convert real suffix to string
                        //     let real_suffix_str = suffix.iter().map(|activity| activity.to_string()).collect::<Vec<_>>().join(", ");

                            // let dl = DamerauLevenshtein::default();
                            // temp_dls += 1.0 - dl.for_str(predicted_suffix_str.as_str(), real_suffix_str.as_str()).nval();
                            // let dls: f64 = 1.0 - dl.for_str(predicted_suffix_str.as_str(), real_suffix_str.as_str()).nval();
                            // println!("Prefix: {:?}, Predicted Suffix: [{}], Real Suffix: [{}], Similarity: {}", prefix, predicted_suffix_str, real_suffix_str, 1.0-dl.for_str(predicted_suffix_str.as_str(), real_suffix_str.as_str()).nval());
                        
                            let dls = damerau_levenshtein_distance(&predicted_suffix, &suffix) as f64;
                            let current_dls;
                            if dls == 0.0 {
                                current_dls = 1.0;
                            }
                            else{
                                current_dls = 1.0 -  dls /max(suffix.len(), predicted_suffix.len()) as f64;
                            }
                            println!("full model trace:{:?}, actual:{:?}, predicted:{:?}, maxlen: {}, current_dls:{}", full_model_trace.clone(), suffix.clone(), predicted_suffix.clone(), max(suffix.len(), predicted_suffix.len()) as f64, current_dls);
                            temp_dls += current_dls;
                            // insert the current_dls to prefix_dls based on its length
                            prefix_dls.entry(prefix.len()).and_modify(|v| *v += current_dls*probability.clone().approx().unwrap()).or_insert(current_dls* probability.clone().approx().unwrap());

                            prefix_prob.entry(prefix.len()).and_modify(|v| *v += probability).or_insert(probability.clone());
                        }
                        None => {
                            println!("no alignment found for prefix {:?}", prefix);
                        },
                    }
                } 
            }
            // println!("correct predictions for this trace: {} / {}", correct_predictions, trace.len());
            // println!("Average similarity for this trace: {}", temp_dls / trace.len() as f64);
            total_dls += temp_dls /trace.len() as f64 * probability.clone().approx().unwrap() as f64;
        }
    // print out prefix_dls based on the order of key
    // let mut keys: Vec<&usize> = prefix_dls.keys().collect();
    // keys.sort();
                    
    // // Print each key and its average
    // for key in keys {
    //     let dls = prefix_dls.get(key).unwrap();                
    //     let prob = prefix_prob.get(key).unwrap().clone().approx().unwrap();
    //     println!("({}, {})", key, *dls / prob as f64);

        Ok(total_dls)
    }
}



pub fn get_next_activity<T, State>(
    semantics: &T,
    prefix: &Vec<Activity>,
    states: Vec<(usize, State)>,
) -> Option<Activity>
where
    T: Semantics<SemState = State> + ?Sized,
    State: Display + Debug + Clone + Hash + Eq,
{
    // log::debug!("transform alignment {:?}", states);
    let mut alignment = vec![];

    let mut it = states.into_iter();

    let (mut previous_trace_index, mut previous_state) = it.next().unwrap();

    let mut counter = 0;
    let mut find_the_next = false;
    let mut next_activity= None;
    for (trace_index, state) in it {
        // log::debug!("transform {} from {} to {}", trace_index, previous_state, state);
        if trace_index != previous_trace_index {
            //we did a move on the log
            let activity = prefix[previous_trace_index];

            if previous_state == state {
                //we did not move on the model => log move
                alignment.push(Move::LogMove(activity));
                counter += 1;
                if counter == prefix.len() {
                    find_the_next = true;
                }

            } else {
                //we moved on the model => synchronous move
                // let transition =
                //     find_transition_with_label(semantics, &previous_state, &state, activity)
                //         .with_context(|| {
                //             format!(
                //                 "Map synchronous move from {} {} to {} {} with label {}",
                //                 previous_trace_index, previous_state, trace_index, state, activity
                //             )
                //         })?;
                // alignment.push(Move::SynchronousMove(activity, transition));
                if find_the_next {
                    next_activity = Some(activity);
                    break;
                }
                counter += 1;
                if counter == prefix.len() {
                    find_the_next = true;
                }
            }
        } else {
            //we did not do a move on the log

            if let Some(transition) =
                is_there_a_silent_transition_enabled(semantics, &previous_state, &state)
            {
                //there is a silent transition enabled, which is the cheapest
                alignment.push(Move::SilentMove(transition));
            } else {
                //otherwise, we take an arbitrary labelled model move
                let transition = find_labelled_transition(semantics, &previous_state, &state);
                // alignment.push(Move::ModelMove(
                //     semantics.get_transition_activity(transition).unwrap(),
                //     transition,
                // ));
                if find_the_next {
                    next_activity = semantics.get_transition_activity(transition.unwrap());
                    break;
                }
            }
        }

        previous_trace_index = trace_index;
        previous_state = state;

        // log::debug!("prefix: {:?}", alignment);
    }
    next_activity
}


pub fn find_suffix_afterwards<T, State>(
    semantics: &T,
    states: Vec<(usize, State)>,
) -> Vec<Activity>
where
    T: Semantics<SemState = State> + ?Sized,
    State: Display + Debug + Clone + Hash + Eq,
{
    let mut it = states.into_iter();

    let (mut previous_trace_index, mut previous_state) = it.next().unwrap();
    let mut suffix2return = Vec::new();
    let mut trace = Vec::new();
    let mut flag = false;

    for (trace_index, state) in it {
        // log::debug!("transform {} from {} to {}", trace_index, previous_state, state);
        if trace_index != previous_trace_index {
            //we did a move on the log
            if previous_state != state {
                // //we moved on the model => synchronous move
                // let transition =
                //     find_transition_with_label(semantics, &previous_state, &state, activity).unwrap();
                flag = true;
            }
        } else {
            //we did not do a move on the log
            if let Some(_) =
                is_there_a_silent_transition_enabled(semantics, &previous_state, &state)
            {
                //there is a silent transition enabled, which is the cheapest
                // alignment.push(Move::SilentMove(transition));
            } 
            else {
                let transition = find_labelled_transition(semantics, &previous_state, &state).unwrap();
                // alignment.push(Move::ModelMove(
                //     semantics.get_transition_activity(transition).unwrap(),
                //     transition,
                // ));
                trace.push(semantics.get_transition_activity(transition).unwrap());
                if flag == true {
                    // add the activity to the suffix
                    suffix2return.push(semantics.get_transition_activity(transition).unwrap());
                }
            }
        }
        previous_trace_index = trace_index;
        previous_state = state;
    }
    suffix2return
}


pub fn find_transition_with_label<T, FS>(
    semantics: &T,
    from: &FS,
    to: &FS,
    label: Activity,
) -> Result<TransitionIndex>
where
    T: Semantics<SemState = FS> + ?Sized,
    FS: Display + Debug + Clone + Hash + Eq,
{
    // log::debug!("find transition with label {}", label);
    for transition in semantics.get_enabled_transitions(from) {
        // log::debug!("transition {} is enabled", transition);
        if semantics.get_transition_activity(transition) == Some(label) {
            let mut from = from.clone();
            semantics.execute_transition(&mut from, transition)?;
            if &from == to {
                return Ok(transition);
            }
        }
    }
    Err(anyhow!(
        "There is no transition with activity {} that brings the model from {} to {}",
        label,
        from,
        to
    ))
}


pub fn is_there_a_silent_transition_enabled<T, FS>(
    semantics: &T,
    from: &FS,
    to: &FS,
) -> Option<TransitionIndex>
where
    T: Semantics<SemState = FS> + ?Sized,
    FS: Display + Debug + Clone + Hash + Eq,
{
    // log::debug!("is there a silent transition enabled from {} to {}", from, to);
    // log::debug!("enabled transitions {:?}", semantics.get_enabled_transitions(from));
    for transition in semantics.get_enabled_transitions(from) {
        if semantics.is_transition_silent(transition) {
            let mut from = from.clone();
            let _ = semantics.execute_transition(&mut from, transition);
            if &from == to {
                // log::debug!("yes");
                return Some(transition);
            }
        }
    }
    // log::debug!("no");
    None
}



pub fn find_labelled_transition<T, FS>(semantics: &T, from: &FS, to: &FS) -> Result<TransitionIndex>
where
    T: Semantics<SemState = FS> + ?Sized,
    FS: Display + Debug + Clone + Hash + Eq,
{
    for transition in semantics.get_enabled_transitions(from) {
        if !semantics.is_transition_silent(transition) {
            let mut from = from.clone();
            semantics.execute_transition(&mut from, transition)?;
            if &from == to {
                return Ok(transition);
            }
        }
    }
    Err(anyhow!(
        "There is no transition with any activity enabled that brings the model from {} to {}",
        from,
        to
    ))
}
// pub fn find_labelled_transition<T, FS>(semantics: &T, from: &FS, to: &FS) -> Result<TransitionIndex>
// where
//     T: Semantics<SemState = FS> + ?Sized,
//     FS: Display + Debug + Clone + Hash + Eq,
// {
//     let mut transition2return = None;
//     let mut count =0;
//     for transition in semantics.get_enabled_transitions(from) {
//         if !semantics.is_transition_silent(transition) {
//             let mut from = from.clone();
//             semantics.execute_transition(&mut from, transition)?;
//             if &from == to {
//                 transition2return = Some(transition);
//                 count +=1;
//             }
//         }
//     }

//     if transition2return.is_some() {
//         return Ok(transition2return.unwrap());
//     }
//     Err(anyhow!(
//         "There is no transition with any activity enabled that brings the model from {} to {}",
//         from,
//         to
//     ))
// }

pub fn damerau_levenshtein_distance<T>(seq1: &[T], seq2: &[T]) -> usize
where
    T: Eq + Hash + Clone,
{
    let len1 = seq1.len();
    let len2 = seq2.len();

    // Create a hashmap to store the last occurrence of each element
    let mut da: HashMap<T, usize> = HashMap::new();

    // Maximum possible distance
    let max_dist = len1 + len2;

    // Create distance matrix with an extra row and column
    let mut h = vec![vec![0usize; len2 + 2]; len1 + 2];

    h[0][0] = max_dist;
    for i in 0..=len1 {
        h[i + 1][0] = max_dist;
        h[i + 1][1] = i;
    }
    for j in 0..=len2 {
        h[0][j + 1] = max_dist;
        h[1][j + 1] = j;
    }

    for i in 1..=len1 {
        let mut db = 0;
        for j in 1..=len2 {
            let k = *da.get(&seq2[j - 1]).unwrap_or(&0);
            let l = db;
            let mut cost = 1;
            if seq1[i - 1] == seq2[j - 1] {
                cost = 0;
                db = j;
            }

            h[i + 1][j + 1] = min(
                min(
                    h[i][j] + cost,              // substitution
                    h[i + 1][j] + 1,             // insertion
                ),
                min(
                    h[i][j + 1] + 1,             // deletion
                    h[k][l] + (i - k - 1) + 1 + (j - l - 1), // transposition
                ),
            );
        }

        da.insert(seq1[i - 1].clone(), i);
    }

    h[len1 + 1][len2 + 1]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_cases() {
        let seq1 = vec!["a0", "a10"];
        let seq2 = vec!["a10", "a11"];
        assert_eq!(damerau_levenshtein_distance(&seq1, &seq2), 2);

        let seq1 = vec!["a0", "a1"];
        let seq2 = vec!["a0", "a1"];
        assert_eq!(damerau_levenshtein_distance(&seq1, &seq2), 0);

        let seq1 = vec!["a0", "a1"];
        let seq2 = vec!["a1", "a0"];
        assert_eq!(damerau_levenshtein_distance(&seq1, &seq2), 1);
    }

    #[test]
    fn test_insertion_deletion() {
        let seq1 = vec!["a0"];
        let seq2 = vec!["a0", "a1"];
        assert_eq!(damerau_levenshtein_distance(&seq1, &seq2), 1);

        let seq1 = vec!["a0", "a1", "a2"];
        let seq2 = vec!["a0", "a2"];
        assert_eq!(damerau_levenshtein_distance(&seq1, &seq2), 1);
    }
}