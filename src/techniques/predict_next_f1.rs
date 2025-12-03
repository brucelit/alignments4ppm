use crate::{
    ebi_framework::displayable::Displayable, ebi_traits::{ebi_trait_finite_stochastic_language::EbiTraitFiniteStochasticLanguage, ebi_trait_stochastic_semantics::EbiTraitStochasticSemantics}, semantics::semantics::Semantics, stochastic_semantics::stochastic_semantics::StochasticSemantics, techniques::{astar_for_prediction},
};
use anyhow::{Result, anyhow};
use ebi_arithmetic::{MaybeExact, Zero, fraction::{fraction::Fraction, fraction_enum::FractionEnum}};
use ebi_objects::{Activity,ebi_objects::labelled_petri_net::TransitionIndex, ActivityKeyTranslator};
use std::{ops::{Add, AddAssign}, str::FromStr};
use std::{
    fmt::{Debug, Display},
    hash::Hash,
    collections::{HashSet,HashMap},
};


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
            balance: self.balance,
            // stochastic_weighted_cost:  (1.0 + cost).ln().powf(ALPHA) * (1.0 - probability.ln()).powf(1.0-ALPHA),
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
        let self_stochastic_cost = self.cost ;
        let other_stochastic_cost = other.cost;
        // let self_stochastic_cost = (1.0 +self.cost).ln() * (1.0 - self.probability.ln());
        // let other_stochastic_cost = (1.0 + other.cost).ln() * (1.0 - other.probability.ln());
        //  let self_stochastic_cost = (((1.0 +self.cost).log10()).powf(self.balance)) * ((1.0 - self.probability.log10()).powf(1.0-self.balance));
        // let other_stochastic_cost = (((1.0 + other.cost).log10()).powf(other.balance)) * ((1.0 - other.probability.log10()).powf(1.0-other.balance));
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

pub trait PredictF1 {
    fn predict_f1(
        &self,
        event_log: &Box<dyn EbiTraitFiniteStochasticLanguage>,
    ) -> Result<f64>;
}

impl PredictF1 for EbiTraitStochasticSemantics {
    fn predict_f1(
        &self,
        event_log: &Box<dyn EbiTraitFiniteStochasticLanguage>,
    ) -> Result<f64> {
        match self {
            EbiTraitStochasticSemantics::Usize(sem) => sem.predict_f1(event_log),
            EbiTraitStochasticSemantics::Marking(sem) => sem.predict_f1(event_log),
            EbiTraitStochasticSemantics::NodeStates(sem) => sem.predict_f1(event_log),
        }
    }
}


impl<State: Displayable> dyn StochasticSemantics<StoSemState = State, SemState = State, AliState = State> {
    pub fn predict_f1(
        &self,
        event_log: &Box<dyn EbiTraitFiniteStochasticLanguage>,
    ) -> Result<f64> {

        let mut first_activity_map: HashSet<Activity> = HashSet::new();
        let mut not_first_activity_map: HashSet<Activity> = HashSet::new();

        // get the percentage of each activity
        let mut activity_percentages_before_normalization: HashMap<Activity, Fraction> = HashMap::new();
        for (trace, probability) in event_log.iter_trace_probability() {
            let mut i = 0;
            for activity in trace {
                if i == 0 {
                    first_activity_map.insert(activity.clone());
                } else {
                    not_first_activity_map.insert(activity.clone());
                }
                i += 1;
                *activity_percentages_before_normalization.entry(activity.clone()).or_insert(Fraction::zero()) += probability;
            }
        }
        // get Hashset difference
        let only_first_activity_map: HashSet<Activity> = first_activity_map
            .difference(&not_first_activity_map)
            .cloned()
            .collect();
        println!("Activities in first activity map: {:?}", first_activity_map);
        println!("Activities in not first activity map: {:?}", not_first_activity_map);
        println!("Activities that only appear as first activity in traces: {:?}", only_first_activity_map);
        // remove these activities from the activity percentages
        for activity in only_first_activity_map {
            activity_percentages_before_normalization.remove(&activity);
        }

        activity_percentages_before_normalization.insert(Activity { id: 999999 as usize }, Fraction::from_str("1.0").unwrap());
        // get the total probability
        let total_probability: Fraction = activity_percentages_before_normalization.values().cloned().sum();

        let mut accuracy = 0.0;
        let balance = 0.89;

        let mut activity_key1 = self.activity_key().clone();
        let translator =
            ActivityKeyTranslator::new(&event_log.activity_key(), &mut activity_key1);

         // initialize the F1-score, precision and recall for each activity
        let mut f1_scores: HashMap<Activity, Fraction> = HashMap::new();
        let mut precisions: HashMap<Activity, Fraction> = HashMap::new();
        let mut recalls: HashMap<Activity, Fraction> = HashMap::new();
        let mut accuracy_each: HashMap<Activity, Fraction> = HashMap::new();

        // initialize the true positive, false positive, false negative for each activity
        let mut true_positives: HashMap<Activity, Fraction> = HashMap::new();
        let mut false_positives: HashMap<Activity, Fraction> = HashMap::new();
        let mut false_negatives: HashMap<Activity, Fraction> = HashMap::new();
        let mut activity_percentages = HashMap::new();

         // normalize the activity percentages and translate to model activity key

        for (activity, prob) in activity_percentages_before_normalization.iter() {
            let normalized_prob = prob / &total_probability;
            if activity.id == 999999 as usize {
                activity_percentages.insert(*activity, normalized_prob);
                continue;
            }
            activity_percentages.insert(translator.translate_activity(&activity), normalized_prob);
        }

        println!("activity percentages: {:?}", activity_percentages);

         for activity in activity_percentages.keys() {
            if activity.id == 999999 as usize {
                continue;
            }
            true_positives.insert(*activity, Fraction::zero());
            false_positives.insert(*activity, Fraction::zero());
            false_negatives.insert(*activity, Fraction::zero());
            accuracy_each.insert(*activity, Fraction::zero());
        }
        true_positives.insert(Activity { id: 999999 as usize }, Fraction::zero());
        false_positives.insert(Activity { id: 999999 as usize }, Fraction::zero());
        false_negatives.insert(Activity { id: 999999 as usize }, Fraction::zero());
        accuracy_each.insert(Activity { id: 999999 as usize }, Fraction::zero());

        let mut accurate_prediction_for_prefix:HashMap<usize,Fraction> = HashMap::new();
        let mut total_prediction_for_prefix:HashMap<usize,Fraction> = HashMap::new();

        for (trace, probability) in event_log.iter_trace_probability() {
            let trace2 = &translator.translate_trace(trace);

            // get the prefix and the next activity
            let mut prefix_collection = Vec::new();
            for i in 1..= trace2.len() {
                prefix_collection.push((trace2[..i].to_vec(), trace2.get(i)));
            }
            
            let mut correct_predictions: i128 = 0;
            
            // iterate the prefix of the trace and align it
            for (prefix, next_activity) in prefix_collection {
                total_prediction_for_prefix.entry(prefix.len()).and_modify(|current_value| *current_value += probability.clone()).or_insert(probability.clone());

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
                                    balance: balance
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
                                        balance: balance
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
                                            balance: balance
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
                                        balance: balance
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
                        let full_model_trace = get_model_trace(self, path.clone());
                        // println!("full model trace: {:?}", full_model_trace);
                        
                        // let predicted_next_activity =  find_model_move_afterwards(self, path);
                        let predicted_next_activity =  find_last_match(&prefix, full_model_trace);


if predicted_next_activity.is_none() && next_activity.is_none() {
    // Both predict and actual are None (end of sequence) - True Positive for "end"
    correct_predictions += 1;

    accurate_prediction_for_prefix.entry(prefix.len())
        .and_modify(|current_value| *current_value += probability.clone()/FractionEnum::from_str(&trace.len().to_string()).unwrap())
        .or_insert(probability.clone()/FractionEnum::from_str(&trace.len().to_string()).unwrap());
    
    true_positives.entry(Activity { id: 999999 as usize })
        .and_modify(|current_value| *current_value += probability.clone()/FractionEnum::from_str(&trace.len().to_string()).unwrap())
        .or_insert(probability.clone()/FractionEnum::from_str(&trace.len().to_string()).unwrap());
}
// Predicted None, but actual has an activity - False Negative for actual activity
else if predicted_next_activity.is_none() && next_activity.is_some() {
    false_negatives.entry(*next_activity.unwrap())
        .and_modify(|current_value| *current_value += probability.clone()/FractionEnum::from_str(&trace.len().to_string()).unwrap())
        .or_insert(probability.clone()/FractionEnum::from_str(&trace.len().to_string()).unwrap());
    
    // Also a False Positive for "end" (we predicted end but it wasn't)
    false_positives.entry(Activity { id: 999999 as usize })
        .and_modify(|current_value| *current_value += probability.clone()/FractionEnum::from_str(&trace.len().to_string()).unwrap())
        .or_insert(probability.clone()/FractionEnum::from_str(&trace.len().to_string()).unwrap());
}
// Predicted an activity, but actual is None - False Positive for predicted activity
else if predicted_next_activity.is_some() && next_activity.is_none() {
    false_positives.entry(predicted_next_activity.unwrap())
        .and_modify(|current_value| *current_value += probability.clone()/FractionEnum::from_str(&trace.len().to_string()).unwrap())
        .or_insert(probability.clone()/FractionEnum::from_str(&trace.len().to_string()).unwrap());
    
    // Also a False Negative for "end" (actual was end but we didn't predict it)
    false_negatives.entry(Activity { id: 999999 as usize })
        .and_modify(|current_value| *current_value += probability.clone()/FractionEnum::from_str(&trace.len().to_string()).unwrap())
        .or_insert(probability.clone()/FractionEnum::from_str(&trace.len().to_string()).unwrap());
}
// Both have activities - check if they match
else if let (Some(predicted), Some(actual)) = (predicted_next_activity, next_activity) {
    if predicted == *actual {
        // True Positive for the correctly predicted activity
        correct_predictions += 1;
        accurate_prediction_for_prefix.entry(prefix.len())
            .and_modify(|current_value| *current_value += probability.clone()/FractionEnum::from_str(&trace.len().to_string()).unwrap())
            .or_insert(probability.clone()/FractionEnum::from_str(&trace.len().to_string()).unwrap());
        
        true_positives.entry(predicted)
            .and_modify(|current_value| *current_value += probability.clone()/FractionEnum::from_str(&trace.len().to_string()).unwrap())
            .or_insert(probability.clone()/FractionEnum::from_str(&trace.len().to_string()).unwrap());
    } else {
        // False Positive for what we predicted
        false_positives.entry(predicted)
            .and_modify(|current_value| *current_value += probability.clone()/FractionEnum::from_str(&trace.len().to_string()).unwrap())
            .or_insert(probability.clone()/FractionEnum::from_str(&trace.len().to_string()).unwrap());
        
        // False Negative for what actually happened
        false_negatives.entry(*actual)
            .and_modify(|current_value| *current_value += probability.clone()/FractionEnum::from_str(&trace.len().to_string()).unwrap())
            .or_insert(probability.clone()/FractionEnum::from_str(&trace.len().to_string()).unwrap());
    }
}
else{
    println!("noch");
}

                            // predicted = actual
        //                     if predicted_next_activity.is_none() && next_activity.is_none() {
        //                         // println!("Correctly predicted the end of prefix {:?}", prefix);
        //                         correct_predictions += 1;
                                
        //                         accurate_prediction_for_prefix.entry(prefix.len()).and_modify(|current_value| *current_value += probability.clone()).or_insert(probability.clone());

        //                         true_positives.entry(Activity { id: 999999 as usize }).and_modify(|current_value| *current_value += probability.clone()).or_insert(probability.clone());
        //                     }

        //                     // predicted is None, actual is not None
        //                     else if predicted_next_activity.is_none() && !next_activity.is_none() {
        //                         false_negatives.entry(*next_activity.unwrap()).and_modify(|current_value| *current_value += probability.clone()).or_insert(probability.clone());
    
        //                     // Also a False Positive for "end" (we predicted end but it wasn't)
        //                     false_positives.entry(Activity { id: 999999 as usize })
        //                         .and_modify(|current_value| *current_value += probability.clone())
        // .or_insert(probability.clone());
        //                         // println!("Incorrectly predicted next activity to be None after prefix {:?}", prefix);
        //                     }
        //                     else if !predicted_next_activity.is_none() && next_activity.is_none() {
        //                         false_negatives.entry(Activity { id: 999999 as usize }).and_modify(|current_value| *current_value += probability.clone()).or_insert(probability.clone());
        //                         // println!("Incorrectly predicted next activity after prefix {:?}, actual next activity is None", prefix);

        //                         false_positives.entry(predicted_next_activity.unwrap())
        //                         .and_modify(|current_value| *current_value += probability.clone())
        //                         .or_insert(probability.clone());
        //                     }
        //                     else if predicted_next_activity.unwrap() == *next_activity.unwrap() {
        //                         correct_predictions += 1;

        //                         accurate_prediction_for_prefix.entry(prefix.len()).and_modify(|current_value| *current_value += probability.clone()).or_insert(probability.clone());

        //                         true_positives.entry(predicted_next_activity.unwrap()).and_modify(|current_value| *current_value += probability.clone()).or_insert(probability.clone());
                            
        //                     } 
        //                     else if let (Some(predicted), Some(actual)) = (predicted_next_activity, next_activity) {
        //                 if predicted == *actual {
        //                     // True Positive for the correctly predicted activity
        //                     correct_predictions += 1;
        //                     accurate_prediction_for_prefix.entry(prefix.len())
        //                         .and_modify(|current_value| *current_value += probability.clone())
        //                         .or_insert(probability.clone());
                            
        //                     true_positives.entry(predicted)
        //                         .and_modify(|current_value| *current_value += probability.clone())
        //                         .or_insert(probability.clone());
        //                 } else {
        //                     // False Positive for what we predicted
        //                     false_positives.entry(predicted)
        //                         .and_modify(|current_value| *current_value += probability.clone())
        //                         .or_insert(probability.clone());
                            
        //                     // False Negative for what actually happened
        //                     false_negatives.entry(*actual)
        //                         .and_modify(|current_value| *current_value += probability.clone())
        //                         .or_insert(probability.clone());
        //                 }
        //             }
        //                     else {
                               
        //                     } 
                        }
                        None => {
                            println!("no alignment found for prefix {:?}", prefix);
                        },
                    }
                } 
            }
            println!("\n correct predictions for this trace: {} / {}, probability: {:?}", correct_predictions, trace.len(), probability.clone().approx().unwrap());
            accuracy += correct_predictions as f64 /trace.len() as f64 * probability.clone().approx().unwrap() as f64;

        }
        // println!("accuracy for prediction: {}", accuracy);
        // println!("accuracy for prefix length 1: {}", accuracy_counter1);
        // println!("accuracy for prefix length 2: {}", accuracy_counter2);
        // println!("accuracy for prefix length 3: {}", accuracy_counter3);
        // println!("accuracy for prefix length 4: {}", accuracy_counter4);
        // println!("Accurate predictions per prefix length: {:?}", accurate_prediction_for_prefix);
        // println!("Total predictions per prefix length: {:?}", total_prediction_for_prefix);

        // divide accurate predictions by total predictions per prefix length
        for (prefix_len, accurate_count) in accurate_prediction_for_prefix.iter() {
            if let Some(total_count) = total_prediction_for_prefix.get(prefix_len) {
                let prefix_accuracy = accurate_count / total_count;
                println!("({}, {})", prefix_len, prefix_accuracy);
            }
        }

        // println!("True Positives: {:?}", true_positives);
        // println!("False Positives: {:?}", false_positives);
        // println!("False Negatives: {:?}", false_negatives);

        let mut tp_total = Fraction::zero();
        let mut fp_total = Fraction::zero();
        let mut fn_total = Fraction::zero();
        let mut prob_total = Fraction::zero();
        let mut accuracy_scores = HashMap::new();
        // print precision, recall and F1-score for each activity
        for activity in activity_percentages.keys() {
            let tp = true_positives.get(activity).unwrap();
            let fp = false_positives.get(activity).unwrap();
            let fn_ = false_negatives.get(activity).unwrap();
            tp_total += tp.clone();
            fp_total += fp.clone();
            fn_total += fn_.clone();
            prob_total += activity_percentages.get(activity).unwrap().clone();

            let precision = if tp + fp > Fraction::zero() {
                tp.clone() / (tp + fp)
            } else {
                Fraction::zero()
            };
            let recall = if tp + fn_ > Fraction::zero() {
                tp.clone() / (tp + fn_)
            } else {
                Fraction::zero()
            };
            let f1_score = if precision.clone() + recall.clone() > Fraction::zero() {
                (precision.clone() * recall.clone() * Fraction::from_str("2")?) / (precision.clone() + recall.clone())
            } else {
                Fraction::zero()
            };

            precisions.insert(activity.clone(), precision.clone());
            recalls.insert(activity.clone(), recall.clone());
            f1_scores.insert(activity.clone(), f1_score.clone());
            accuracy_scores.insert(activity.clone(), (tp.clone()) / (tp.clone() + fp.clone() + fn_.clone()));
        }
        // get the weighted f1 score
        let mut weighted_f1_score = Fraction::zero();
        let mut weighted_acc_score = Fraction::zero();
        let mut total_f1_score = Fraction::zero();

        for (activity, f1_score) in f1_scores.iter() {
            let activity_percentage = activity_percentages.get(activity).unwrap();
            weighted_f1_score += f1_score.clone() * activity_percentage.clone();
            total_f1_score += f1_score.clone();
        }

        for (activity, acc_score) in accuracy_scores.iter() {
            let activity_percentage = activity_percentages.get(activity).unwrap();
            weighted_acc_score += acc_score.clone() * activity_percentage.clone();
        }

        println!("total probability sum check: {:?}", prob_total.approx().unwrap());
        println!("Weighted F1-Score: {:.4}", weighted_f1_score.approx().unwrap());
        println!("Macro F1-Score: {:.4}", (total_f1_score / Fraction::from_str(&activity_percentages.len().to_string())?).approx().unwrap());
        println!("sum tp: {:?}, fp: {:?}, fn: {:?}", tp_total.clone().approx().unwrap(), fp_total.clone().approx().unwrap(), fn_total.clone().approx().unwrap());

        println!("Final accuracy for prediction: {}",tp_total.clone().approx().unwrap() / (tp_total + fp_total).approx().unwrap());
        Ok(accuracy)

    }
}

pub fn find_model_move_afterwards<T, State>(
    semantics: &T,
    states: Vec<(usize, State)>,
) -> Option<Activity>
where
    T: Semantics<SemState = State> + ?Sized,
    State: Display + Debug + Clone + Hash + Eq,
{
    let mut it = states.into_iter();

    let (mut previous_trace_index, mut previous_state) = it.next().unwrap();
    let mut activity2return = None;
    let mut flag = false;

    for (trace_index, state) in it {
        // log::debug!("transform {} from {} to {}", trace_index, previous_state, state);
        if trace_index != previous_trace_index {

            if previous_state != state {
                // //we moved on the model => synchronous move
                // let transition =
                //     find_transition_with_label(semantics, &previous_state, &state, activity).unwrap();
                flag = true;
                activity2return = None;
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
                if flag == true {
                    activity2return = semantics.get_transition_activity(transition);
                    flag = false;
                }
            }
        }
        previous_trace_index = trace_index;
        previous_state = state;

        // log::debug!("prefix: {:?}", alignment);
    }
    activity2return
}

pub fn get_model_trace<T, State>(
    semantics: &T,
    states: Vec<(usize, State)>,
) -> Vec<Activity>
where
    T: Semantics<SemState = State> + ?Sized,
    State: Display + Debug + Clone + Hash + Eq,
{
    let mut it = states.into_iter();

    let (mut previous_trace_index, mut previous_state) = it.next().unwrap();
    let mut trace2return = Vec::new();

    for (trace_index, state) in it {
        // log::debug!("transform {} from {} to {}", trace_index, previous_state, state);
         if trace_index != previous_trace_index {

            if previous_state != state {
                // //we moved on the model => synchronous move
                // let transition =
                //     find_transition_with_label(semantics, &previous_state, &state, activity).unwrap();
                let transition = find_labelled_transition(semantics, &previous_state, &state).unwrap();
                trace2return.push(semantics.get_transition_activity(transition).unwrap());
            }
        } else {
            //we did not do a move on the trace
            if let Some(_) =
                is_there_a_silent_transition_enabled(semantics, &previous_state, &state)
            {
                //there is a silent transition enabled, which is the cheapest
                // alignment.push(Move::SilentMove(transition));
            } 
            else {
                let transition = find_labelled_transition(semantics, &previous_state, &state).unwrap();
                trace2return.push(semantics.get_transition_activity(transition).unwrap());
            }
        }
        previous_trace_index = trace_index;
        previous_state = state;
    }
    trace2return
}

pub fn find_last_match(s1: &Vec<Activity>, s2: Vec<Activity>) -> Option<Activity>
{
    let m = s1.len();
    let n = s2.len();

    if m == 0 || n == 0 {
        return None;
    }

    // --- 1. Fill the DP Table (Same as before) ---
    let mut dp = vec![vec![0; n + 1]; m + 1];

    for i in 0..=m {
        dp[i][0] = i;
    }
    for j in 0..=n {
        dp[0][j] = j;
    }

    for i in 1..=m {
        for j in 1..=n {
            let cost = if s1[i - 1] == s2[j - 1] { 0 } else { 1 };

            let match_mismatch = dp[i - 1][j - 1] + cost;
            let deletion = dp[i - 1][j] + 1;
            let insertion = dp[i][j - 1] + 1;

            dp[i][j] = match_mismatch.min(deletion).min(insertion);
        }
    }

    // --- 2. Traceback (NEW PRIORITY) ---

    let mut i = m;
    let mut j = n;

    while i > 0 && j > 0 {
        let current_cost = dp[i][j];
        let is_match = s1[i - 1] == s2[j - 1];
        let match_cost = if is_match { 0 } else { 1 };

        let match_mismatch_path_cost = dp[i - 1][j - 1] + match_cost;
        let deletion_path_cost = dp[i - 1][j] + 1;
        let insertion_path_cost = dp[i][j - 1] + 1;

        // --- This is the core logic ---

        // Check if the current optimal path *could* be a match
        if is_match && current_cost == match_mismatch_path_cost {
            // It is a match. Is it on our *prioritized* path?
            // We must check if the other paths (Delete/Insert)
            // *also* led to this same cost. If they did, we
            // must follow their priority first.
            
            // If deletion is NOT a tied path, and insertion is NOT a tied path,
            // then this match is our only (or highest-priority) choice.
            if current_cost != deletion_path_cost && current_cost != insertion_path_cost {
                //  return Some((i - 1, j - 1));
                if j == s2.len(){
                    return None;
                }
                else
                {
                    return Some(s2[j]);
                }
            }
            // If there *is* a tie, we fall through to the priority
            // logic below. The match will be found later in the
            // traceback once the higher-priority moves are made.
        }

        // 1. Prioritize Deletion (vertical move)
        if current_cost == deletion_path_cost {
            i -= 1;
        }
        // 2. Then prioritize Insertion (horizontal move)
        else if current_cost == insertion_path_cost {
            j -= 1;
        }
        // 3. Finally, take the Match/Mismatch (diagonal move)
        else if current_cost == match_mismatch_path_cost {
            // If we are here, it means this was the only path,
            // or it was a tie and the other paths were not chosen.
            
            // If it was a true match, we must have found it already.
            // But if we are here, it must be a Mismatch.
            // Or, it was a Match that was *not* the last one.
            // (e.g., 'a' at (0,0) is found after this)
            // We just need to trace back.
            
            // But what if the *last match* is the one being
            // de-prioritized in a tie?
            // Let's re-think this...

            // --- Simplified Traceback Logic ---
            // Let's just find the last match on the *chosen* path.
            
            i -= 1;
            j -= 1;
        } else {
            unreachable!("DP traceback logic error");
        }
    }

    // If the loop finishes, we need to check the first match
    // (which is the last one found by the traceback)
    // The previous logic is flawed; let's correct it.
    // We will find the *full* path first, then find the last match.

    // --- 2. Traceback (Corrected Logic) ---

    // We will store the last match found *on the prioritized path*.
    let mut last_match_indices: Option<Activity> = None;
    i = m;
    j = n;
    
    while i > 0 && j > 0 {
        let current_cost = dp[i][j];
        let is_match = s1[i - 1] == s2[j - 1];
        let match_cost = if is_match { 0 } else { 1 };

        let match_mismatch_path_cost = dp[i - 1][j - 1] + match_cost;
        let deletion_path_cost = dp[i - 1][j] + 1;
        let insertion_path_cost = dp[i][j - 1] + 1;
        
        // --- This is the changed priority ---

        // 1. Prioritize Deletion (vertical move)
        if current_cost == deletion_path_cost {
            i -= 1;
        } 
        // 2. Then prioritize Insertion (horizontal move)
        else if current_cost == insertion_path_cost {
            j -= 1;
        } 
        // 3. Finally, take the Match/Mismatch (diagonal move)
        else if current_cost == match_mismatch_path_cost {
            if is_match && last_match_indices.is_none() {
                // This is the first match we've found on our
                // prioritized path, so it's the *last* match.
                if j == s2.len(){
                    last_match_indices = None;
                }
                else
                {
                    last_match_indices = Some(s2[j]);
                }
                // last_match_indices = Some((i - 1, j - 1));
            }
            i -= 1;
            j -= 1;
        } else {
            unreachable!("DP traceback logic error");
        }
    }
    if s1.len() == 1 && last_match_indices.is_none(){
        return Some(s2[0]);
    }

    return last_match_indices;
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

// impl<State: Displayable> dyn StochasticSemantics<StoSemState = State, SemState = State, AliState = State> {
//     pub fn predict_f1_with_fixed_prefix_length(
//         &self,
//         prefix_len: usize,
//         event_log: &Box<dyn EbiTraitFiniteStochasticLanguage>,
//     ) -> Result<f64> {
//         let mut accuracy = 0.0;

//         let mut activity_key1 = self.activity_key().clone();
//         let translator =
//             ActivityKeyTranslator::new(&event_log.activity_key(), &mut activity_key1);

//         let mut probability_from_test_to_count = 0.0;
//         let mut probability_from_test_to_not_count = 0.0;
//         let mut right_probability_sum = 0.0;

//         for (trace, probability) in event_log.iter_trace_probability() {
//             let trace2 = &translator.translate_trace(trace);


//             // get the prefix and the next activity
//             let mut prefix_collection = Vec::new();

//             if trace2.len() < prefix_len {
//                 probability_from_test_to_not_count += probability.clone().approx().unwrap() as f64;
//                 continue;
//             }

//             probability_from_test_to_count += probability.clone().approx().unwrap() as f64;

//             prefix_collection.push((trace2[0..prefix_len].to_vec(), trace2.get(prefix_len)));

//             let mut correct_predictions: i128 = 0;
            
//             // iterate the prefix of the trace and align it
//             for (prefix, next_activity) in prefix_collection {
//                 // get the start state
//                 if let Some(initial_state) = self.get_initial_state() {
//                     let start = (0, initial_state);

//                     // successor relation in the model
//                     let successors = |(trace_index, state): &(usize, State)| {
//                         let mut result: Vec<((usize, State), StochasticWeightedCost)> = vec![];

//                         // log::debug!("successors of log {} model {}", trace_index, state);
//                         if trace_index < &prefix.len() {
//                             //we can do a log move
//                             // log::debug!("\tlog move {}", trace[*trace_index]);

//                             result.push((
//                                 (trace_index + 1, state.clone()),
//                                 StochasticWeightedCost {
//                                     cost: 1.0,
//                                     probability: 1.0,
//                                     // stochastic_weighted_cost: 0.0
//                                 },
//                             ));
//                         }

//                         //walk through the enabled transitions in the model
//                         for transition in self.get_enabled_transitions(&state) {
//                             let total_weight = self
//                                 .get_total_weight_of_enabled_transitions(&state)
//                                 .unwrap();

//                             let mut new_state = state.clone();
//                             // log::debug!("\t\tnew state before {}", new_state);
//                             let _ = self.execute_transition(&mut new_state, transition);
//                             // log::debug!("\t\tnew state after {}", new_state);

//                             let transition_weight = self.get_transition_weight(&state, transition);
//                             let transition_probability: f64 = (transition_weight / &total_weight).approx().unwrap();

//                             if let Some(activity) = self.get_transition_activity(transition) {
//                                 //non-silent model move
//                                 result.push((
//                                     (*trace_index, new_state.clone()),
//                                     StochasticWeightedCost {
//                                         cost: 1.0,
//                                         probability: transition_probability,
//                                         // stochastic_weighted_cost: 0.0
//                                     },
//                                 ));
//                                 // log::debug!("\tmodel move t{} {} to {}", transition, activity, new_state);

//                                 //which may also be a synchronous move
//                                 if trace_index < &prefix.len() && activity == prefix[*trace_index] {
//                                     //synchronous move
//                                     // log::debug!("\tsynchronous move t{} {} to {}", transition, activity, new_state);
//                                     result.push((
//                                         (trace_index + 1, new_state),
//                                         StochasticWeightedCost {
//                                             cost: 0.0,
//                                             probability: transition_probability,
//                                             // stochastic_weighted_cost: 0.0
//                                         },
//                                     ));
//                                 }
//                             } else {
//                                 //silent move
//                                 result.push((
//                                     (*trace_index, new_state),
//                                     StochasticWeightedCost {
//                                         cost: 0.0,
//                                         probability: transition_probability,
//                                         // stochastic_weighted_cost: 0.0
//                                     },
//                                 ));
//                             }
//                         }

//                         // log::debug!("successors of {} {}: {:?}", trace_index, state, result);
//                         result
//                     };

//                     //function that returns a heuristic on how far we are still minimally from a final state
//                     let heuristic = |_astate: &(usize, State)| StochasticWeightedCost::zero();

//                     //function that returns whether we are in a final synchronous product state
//                     let success = |(trace_index, state): &(usize, State)| {
//                         trace_index == &prefix.len() && self.is_final_state(&state)
//                     };
//                     match astar_for_prediction::astar(&start, successors, heuristic, success) {
//                         Some((path, _cost)) => {
//                             let predicted_next_activity =  find_model_move_afterwards(self, &prefix, path);
//                             if predicted_next_activity.is_none() && next_activity.is_none() {
//                                 right_probability_sum += probability.clone().approx().unwrap() as f64;
//                                 // println!("Correctly predicted the end of prefix {:?}", prefix);
//                                 correct_predictions += 1;
//                             }
//                             else if predicted_next_activity.is_none() && !next_activity.is_none() {
//                                 // println!("Incorrectly predicted next activity to be None after prefix {:?}", prefix);
//                             }
//                             else if !predicted_next_activity.is_none() && next_activity.is_none() {
//                                 // println!("Incorrectly predicted next activity after prefix {:?}, actual next activity is None", prefix);
//                             }
//                             else if predicted_next_activity.unwrap() == *next_activity.unwrap() {
//                                 correct_predictions += 1;
//                                 right_probability_sum += probability.clone().approx().unwrap() as f64 ;
//                                 // println!("Correctly predicted next activity: {:?} after prefix {:?}", predicted_next_activity, prefix);
//                             } else {
//                                 // println!("Incorrectly predicted next activity {:?} to be after prefix {:?}, actual next activity is {:?}", predicted_next_activity, prefix, next_activity);
//                             } 
//                         }
//                         None => {
//                             println!("no alignment found for prefix {:?}", prefix);
//                         },
//                     }
//                 } 
//             }
//             accuracy += correct_predictions as f64  / trace.len() as f64 * probability.clone().approx().unwrap() as f64;
//         }
//         println!("prefix len: {}", prefix_len);
//         println!("accuracy for prediction: {}", accuracy);
//         println!("probability to count from test log: {}", probability_from_test_to_count);
//         println!("probability to not count from test log: {}", probability_from_test_to_not_count);
//         println!("probability for rightly prediction from test log: {}", right_probability_sum);
//         Ok(accuracy)
//     }
// }

// impl<State: Displayable> dyn StochasticSemantics<StoSemState = State, SemState = State, AliState = State> {
//     pub fn predict_f1_with_large_prefix_length(
//         &self,
//         prefix_len: usize,
//         event_log: Box<dyn EbiTraitFiniteStochasticLanguage>,
//     ) -> Result<f64> {
//         let mut accuracy = 0.0;

//         let mut activity_key1 = self.activity_key().clone();
//         let translator =
//             ActivityKeyTranslator::new(&event_log.activity_key(), &mut activity_key1);
        
//         let mut probability_from_test_to_count = 0.0;
//         let mut probability_from_test_to_not_count = 0.0;
//         let mut right_probability_sum = 0.0;

//         for (trace, probability) in event_log.iter_trace_probability() {
//             let trace2 = &translator.translate_trace(trace);

//             if trace2.len() > 10 {
//                 continue;
//             }

//             if trace2.len() < prefix_len {
//                 probability_from_test_to_not_count += probability.clone().approx().unwrap() as f64;
//                 continue;
//             }
//             probability_from_test_to_count += probability.clone().approx().unwrap() as f64;

//             // get the prefix and the next activity
//             let mut prefix_collection = Vec::new();
//             // for i in prefix_len..= trace2.len() {
//             //     prefix_collection.push((trace2[..i].to_vec(), trace2.get(i)));
//             //     }

//             // if trace2.len() == prefix_len {
//             //     prefix_collection.push((trace2.to_vec(), None));
//             // }
//             // else {
//             for i in prefix_len..= trace2.len() {
//                 prefix_collection.push((trace2[0..i].to_vec(), trace2.get(i)));
//             }
//             // }

//             let mut correct_predictions: i128 = 0;
            
//             // iterate the prefix of the trace and align it
//             for (prefix, next_activity) in prefix_collection {
//                 // get the start state
//                 if let Some(initial_state) = self.get_initial_state() {
//                     let start = (0, initial_state);

//                     // successor relation in the model
//                     let successors = |(trace_index, state): &(usize, State)| {
//                         let mut result: Vec<((usize, State), StochasticWeightedCost)> = vec![];

//                         // log::debug!("successors of log {} model {}", trace_index, state);
//                         if trace_index < &prefix.len() {
//                             //we can do a log move
//                             // log::debug!("\tlog move {}", trace[*trace_index]);

//                             result.push((
//                                 (trace_index + 1, state.clone()),
//                                 StochasticWeightedCost {
//                                     cost: 1.0,
//                                     probability: 1.0,
//                                     // stochastic_weighted_cost: 0.0
//                                 },
//                             ));
//                         }

//                         //walk through the enabled transitions in the model
//                         for transition in self.get_enabled_transitions(&state) {
//                             let total_weight = self
//                                 .get_total_weight_of_enabled_transitions(&state)
//                                 .unwrap();

//                             let mut new_state = state.clone();
//                             // log::debug!("\t\tnew state before {}", new_state);
//                             let _ = self.execute_transition(&mut new_state, transition);
//                             // log::debug!("\t\tnew state after {}", new_state);

//                             let transition_weight = self.get_transition_weight(&state, transition);
//                             let transition_probability: f64 = (transition_weight / &total_weight).approx().unwrap();

//                             if let Some(activity) = self.get_transition_activity(transition) {
//                                 //non-silent model move
//                                 result.push((
//                                     (*trace_index, new_state.clone()),
//                                     StochasticWeightedCost {
//                                         cost: 1.0,
//                                         probability: transition_probability,
//                                         // stochastic_weighted_cost: 0.0
//                                     },
//                                 ));
//                                 // log::debug!("\tmodel move t{} {} to {}", transition, activity, new_state);

//                                 //which may also be a synchronous move
//                                 if trace_index < &prefix.len() && activity == prefix[*trace_index] {
//                                     //synchronous move
//                                     // log::debug!("\tsynchronous move t{} {} to {}", transition, activity, new_state);
//                                     result.push((
//                                         (trace_index + 1, new_state),
//                                         StochasticWeightedCost {
//                                             cost: 0.0,
//                                             probability: transition_probability,
//                                             // stochastic_weighted_cost: 0.0
//                                         },
//                                     ));
//                                 }
//                             } else {
//                                 //silent move
//                                 result.push((
//                                     (*trace_index, new_state),
//                                     StochasticWeightedCost {
//                                         cost: 0.0,
//                                         probability: transition_probability,
//                                         // stochastic_weighted_cost: 0.0
//                                     },
//                                 ));
//                             }
//                         }

//                         // log::debug!("successors of {} {}: {:?}", trace_index, state, result);
//                         result
//                     };

//                     //function that returns a heuristic on how far we are still minimally from a final state
//                     let heuristic = |_astate: &(usize, State)| StochasticWeightedCost::zero();

//                     //function that returns whether we are in a final synchronous product state
//                     let success = |(trace_index, state): &(usize, State)| {
//                         trace_index == &prefix.len() && self.is_final_state(&state)
//                     };
//                     match astar_for_prediction::astar(&start, successors, heuristic, success) {
//                         Some((path, _cost)) => {
//                             let predicted_next_activity =  find_model_move_afterwards(self, &prefix, path);
//                             if predicted_next_activity.is_none() && next_activity.is_none() {
//                                 // println!("Correctly predicted the end of prefix {:?}", prefix);
//                                 right_probability_sum += probability.clone().approx().unwrap() as f64 / trace.len() as f64;
//                                 correct_predictions += 1;
//                             }
//                             else if predicted_next_activity.is_none() && !next_activity.is_none() {
//                                 // println!("Incorrectly predicted next activity to be None after prefix {:?}", prefix);
//                             }
//                             else if !predicted_next_activity.is_none() && next_activity.is_none() {
//                                 // println!("Incorrectly predicted next activity after prefix {:?}, actual next activity is None", prefix);
//                             }
//                             else if predicted_next_activity.unwrap() == *next_activity.unwrap() {
//                                 correct_predictions += 1;
//                                 right_probability_sum += probability.clone().approx().unwrap() as f64 / trace.len() as f64;
//                                 // println!("Correctly predicted next activity: {:?} after prefix {:?}", predicted_next_activity, prefix);
//                             } else {
//                                 // println!("Incorrectly predicted next activity {:?} to be after prefix {:?}, actual next activity is {:?}", predicted_next_activity, prefix, next_activity);
//                             } 
//                         }
//                         None => {
//                             println!("no alignment found for prefix {:?}", prefix);
//                         },
//                     }
//                 } 
//             }
//             // println!("probability of this trace: {}", probability);
//             // println!("trace: {:?}", trace2);
//             accuracy += correct_predictions as f64 / trace.len() as f64 * probability.clone().approx().unwrap() as f64;
//         }
//         println!("prefix len: {}", prefix_len);
//         println!("accuracy for prediction: {}", accuracy);
//         println!("probability to count: {}", probability_from_test_to_count);
//         println!("probability to not count: {}", probability_from_test_to_not_count);
//         println!("right sum: {}", right_probability_sum);
//         Ok(accuracy)

//     }
// }
