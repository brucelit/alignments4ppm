use std::collections::{HashMap};
use std::time::{SystemTime, UNIX_EPOCH};
use anyhow::{Context, anyhow};
use clap::{Arg, ArgAction, value_parser};
use ebi_objects::{activity_key::activity::Activity, HasActivityKey, FiniteStochasticLanguage};
use ebi_arithmetic::{Fraction, Zero,fraction::{fraction_enum::FractionEnum}};
use crate::{
    ebi_framework::{
        ebi_command::EbiCommand,
        ebi_input::EbiInputType,
        ebi_output::{EbiOutput, EbiOutputType},
        ebi_trait::EbiTrait,
    }, ebi_traits::{
        ebi_trait_finite_stochastic_language::EbiTraitFiniteStochasticLanguage,
        ebi_trait_stochastic_semantics::{EbiTraitStochasticSemantics, ToStochasticSemantics},
    }, math::constant_fraction::ConstFraction, techniques::{predict_next_activity::PredictTrace, predict_next_test_balance::PredictNextActivityTestBalance,
        predict_next_f1::PredictF1,
        predict_suffix::PredictSuffix, predict_suffix_test_balance::PredictSuffixTestBalance}
};


pub const EBI_PREDICTION: EbiCommand = EbiCommand::Group {
    name_short: "pred",
    name_long: Some("prediction"),
    explanation_short: "Predict the next sequence of events in a running trace.",
    explanation_long: None,
    children: &[
        &EBI_PREDICTION_TRACE_NEXT,
        &EBI_PREDICTION_NEXT_ACTIVITY,
        &EBI_PREDICTION_NEXT_ACTIVITY_TEST_BALANCE,
        &EBI_PREDICTION_SUFFIX,
        &EBI_PREDICTION_SUFFIX_TEST_BALANCE,
        &EBI_PREDICTION_NEXT_ACTIVITY_F1,
    ],
};

pub const EBI_PREDICTION_TRACE_NEXT: EbiCommand = EbiCommand::Command {
    name_short: "pretra",
    name_long: Some("predict-trace"),
    library_name: "ebi_commands::ebi_command_prediction::EBI_PREDICT_TRACE",
    explanation_short: "Compute the most likely next activity of a trace given the stochastic model.",
    explanation_long: None,
    latex_link: None,
    cli_command: Some(|command| {
        command.arg(
            Arg::new("trace")
                .action(ArgAction::Set)
                .value_name("TRACE")
                .help("The trace.")
                .required(true)
                .value_parser(value_parser!(String))
                .num_args(0..),
        )
    }),
    exact_arithmetic: false,
    input_types: &[
        &[&EbiInputType::Trait(EbiTrait::StochasticSemantics)],
        &[&EbiInputType::Fraction(
            Some(ConstFraction::zero()),
            Some(ConstFraction::one()),
            None,
        )],
    ],
    input_names: &["FILE", "VALUE"],
    input_helps: &[
        "The model.",
        "Balance between 0 (=only consider deviations) to 1 (=only consider weight in the model)",
    ],
    execute: |mut inputs, cli_matches| {
        let mut semantics: Box<EbiTraitStochasticSemantics> = inputs.remove(0).to_type::<EbiTraitStochasticSemantics>()?;
        // let balance = inputs.remove(0).to_type::<Fraction>()?;
        if let Some(x) = cli_matches.unwrap().get_many::<String>("trace") {
            let t: Vec<&String> = x.collect();
            let trace = t
                .into_iter()
                .map(|activity| activity.as_str())
                .collect::<Vec<_>>();
            let trace = semantics.activity_key_mut().process_trace_ref(&trace);

            log::trace!("predict the trace {:?} given the model", trace);

            let num = semantics
                .predict_trace(&trace)
                .with_context(|| format!("cannot explain the trace {:?}", trace))?;
            Ok(EbiOutput::Fraction(FractionEnum::from((num, trace.len()))))
        } else {
            Err(anyhow!("no trace given"))
        }
    },
    output_type: &EbiOutputType::Fraction,
};


pub const EBI_PREDICTION_NEXT_ACTIVITY: EbiCommand = EbiCommand::Command {
    name_short: "prenext",
    name_long: Some("predict-next-activity"),
    library_name: "ebi_commands::ebi_command_prediction::EBI_PREDICT_NEXT_ACTIVITY",
    explanation_short: "Compute the most likely next activity of a log given the stochastic model.",
    explanation_long: None,
    latex_link: None,
    cli_command: None,
    exact_arithmetic: false,
    input_types: &[
        &[&EbiInputType::Trait(EbiTrait::FiniteStochasticLanguage)],
        &[&EbiInputType::Trait(EbiTrait::StochasticSemantics)],
    ],
    input_names: &["FILE_1", "FILE_2"],
    input_helps: &[
        "The test data.",
        "The training data."
    ],
    execute: |mut inputs, _| {
        let start = SystemTime::now();
        let test_log = inputs
        .remove(0).to_type::<dyn EbiTraitFiniteStochasticLanguage>()?;
        
        let semantics = inputs.remove(0).to_type::<EbiTraitStochasticSemantics>()?;

        // // for prefix of length of length 2
        let accuracy = semantics
            .predict_next_activity(&test_log)
            .context("cannot make prediction for the log")?;
        let since_the_epoch = start
        .duration_since(UNIX_EPOCH)
        .expect("time should go forward");
        println!("{:?}", since_the_epoch.as_millis());
        Ok(EbiOutput::Fraction(Fraction::zero()))
    },
    output_type: &EbiOutputType::Fraction,
};

pub const EBI_PREDICTION_NEXT_ACTIVITY_F1: EbiCommand = EbiCommand::Command {
    name_short: "pref1",
    name_long: Some("predict-f1"),
    library_name: "ebi_commands::ebi_command_prediction::EBI_PREDICT_NEXT_ACTIVITY",
    explanation_short: "Compute the most likely next activity of a log given the stochastic model.",
    explanation_long: None,
    latex_link: None,
    cli_command: None,
    exact_arithmetic: false,
    input_types: &[
        &[&EbiInputType::Trait(EbiTrait::FiniteStochasticLanguage)],
        &[&EbiInputType::Trait(EbiTrait::StochasticSemantics)],
    ],
    input_names: &["FILE_1", "FILE_2"],
    input_helps: &[
        "The test data.",
        "The training data."
    ],
    execute: |mut inputs, _| {
        let start = SystemTime::now();
        let test_log = inputs
        .remove(0).to_type::<dyn EbiTraitFiniteStochasticLanguage>()?;
        
        let semantics = inputs.remove(0).to_type::<EbiTraitStochasticSemantics>()?;

        // // for prefix of length of length 2
        let accuracy = semantics
            .predict_f1(&test_log)
            .context("cannot make prediction for the log")?;
        println!("Overall accuracy: {}", accuracy);
        let since_the_epoch = start
        .duration_since(UNIX_EPOCH)
        .expect("time should go forward");
        println!("{:?}", since_the_epoch.as_millis());
        Ok(EbiOutput::Fraction(Fraction::zero()))
    },
    output_type: &EbiOutputType::Fraction,
};

pub const EBI_PREDICTION_NEXT_ACTIVITY_TEST_BALANCE: EbiCommand = EbiCommand::Command {
    name_short: "prenext_test_balance",
    name_long: Some("predict-next-activity-test-balance"),
    library_name: "ebi_commands::ebi_command_prediction::EBI_PREDICT_NEXT_ACTIVITY_TEST_BALANCE",
    explanation_short: "Compute the most likely next activity of a log given the stochastic model.",
    explanation_long: None,
    latex_link: None,
    cli_command: None,
    exact_arithmetic: false,
    input_types: &[
        &[&EbiInputType::Trait(EbiTrait::FiniteStochasticLanguage)],
        &[&EbiInputType::Trait(EbiTrait::StochasticSemantics)],
    ],
    input_names: &["FILE_1", "FILE_2"],
    input_helps: &[
        "The test data.",
        "The training data."
    ],
    execute: |mut inputs, _| {
        let start = SystemTime::now();
        let test_log = inputs
        .remove(0).to_type::<dyn EbiTraitFiniteStochasticLanguage>()?;
        
        let semantics = inputs.remove(0).to_type::<EbiTraitStochasticSemantics>()?;

        // // for prefix of length of length 2
        let accuracy = semantics
            .predict_next_activity_test_balance(&test_log)
            .context("cannot make prediction for the log")?;
        println!("Overall accuracy: {}", accuracy);
        let since_the_epoch = start
        .duration_since(UNIX_EPOCH)
        .expect("time should go forward");
        println!("{:?}", since_the_epoch.as_millis());
        Ok(EbiOutput::Fraction(Fraction::zero()))
    },
    output_type: &EbiOutputType::Fraction,
};

pub const EBI_PREDICTION_SUFFIX: EbiCommand = EbiCommand::Command {
    name_short: "presfx",
    name_long: Some("predict-suffix"),
    library_name: "ebi_commands::ebi_command_prediction::EBI_PREDICT_SUFFIX",
    explanation_short: "Compute the most likely suffix.",
    explanation_long: None,
    latex_link: None,
    cli_command: None,
    exact_arithmetic: false,
    input_types: &[
        &[&EbiInputType::Trait(EbiTrait::FiniteStochasticLanguage)],
        &[&EbiInputType::Trait(EbiTrait::StochasticSemantics)],
    ],
    input_names: &["FILE_1", "FILE_2"],
    input_helps: &[
        "The test data.",
        "The training data."
    ],
    execute: |mut inputs, _| {
        let start = SystemTime::now();
        let test_log = inputs
        .remove(0).to_type::<dyn EbiTraitFiniteStochasticLanguage>()?;
        
        let semantics = inputs.remove(0).to_type::<EbiTraitStochasticSemantics>()?;

        // // for prefix of length of length 2
        let accuracy = semantics
            .predict_suffix(&test_log)
            .context("cannot make prediction for the log")?;
        println!("Overall dls: {}", accuracy);
        let since_the_epoch = start
        .duration_since(UNIX_EPOCH)
        .expect("time should go forward");
        println!("{:?}", since_the_epoch.as_millis());
        Ok(EbiOutput::Fraction(Fraction::zero()))
    },
    output_type: &EbiOutputType::Fraction,
};

pub const EBI_PREDICTION_SUFFIX_TEST_BALANCE: EbiCommand = EbiCommand::Command {
    name_short: "presfx_test_balance",
    name_long: Some("predict-suffix-test-balance"),
    library_name: "ebi_commands::ebi_command_prediction::EBI_PREDICTION_SUFFIX_TEST_BALANCE",
    explanation_short: "Compute the most likely suffix.",
    explanation_long: None,
    latex_link: None,
    cli_command: None,
    exact_arithmetic: false,
    input_types: &[
        &[&EbiInputType::Trait(EbiTrait::FiniteStochasticLanguage)],
        &[&EbiInputType::Trait(EbiTrait::StochasticSemantics)],
    ],
    input_names: &["FILE_1", "FILE_2"],
    input_helps: &[
        "The test data.",
        "The training data."
    ],
    execute: |mut inputs, _| {
        let start = SystemTime::now();
        let test_log = inputs
        .remove(0).to_type::<dyn EbiTraitFiniteStochasticLanguage>()?;
        
        let semantics = inputs.remove(0).to_type::<EbiTraitStochasticSemantics>()?;

        // // for prefix of length of length 2
        let accuracy = semantics
            .predict_suffix_test_balance(&test_log)
            .context("cannot make prediction for the log")?;
        println!("Overall dls: {}", accuracy);
        let since_the_epoch = start
        .duration_since(UNIX_EPOCH)
        .expect("time should go forward");
        println!("{:?}", since_the_epoch.as_millis());
        Ok(EbiOutput::Fraction(Fraction::zero()))
    },
    output_type: &EbiOutputType::Fraction,
};


pub fn filter_traces_by_length(log: &Box<dyn EbiTraitFiniteStochasticLanguage>, length: &usize) -> Box<EbiTraitStochasticSemantics> {
    let mut filtered_traces: HashMap<Vec<Activity>, Fraction> = HashMap::new();
    for (trace, probability) in log.iter_trace_probability(){
        if trace.len() >= *length {
            filtered_traces.insert(trace.clone(), probability.clone());
        }
    }
    let mut filtered_log = FiniteStochasticLanguage::new_raw(filtered_traces,log.activity_key().clone());
    filtered_log.normalise();
    // println!("trace length being considered: {}", length);
    println!("preffiltered log: {:?}", filtered_log.traces.len());
    Box::new(filtered_log.to_stochastic_semantics())
}




