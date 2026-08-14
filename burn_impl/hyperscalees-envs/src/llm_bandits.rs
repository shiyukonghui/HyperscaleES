//! LLM bandit task pure scoring logic, ported from
//! `src/hyperscalees/environments/llm_bandits.py`.
//!
//! Only the pure string / regex scoring helpers are ported here. The
//! dataset-loading / external-library pieces (`datasets`, `reasoning_gym`,
//! `math_verify`, tokenizers, `@lru_cache`) and the `Task` subclasses are out
//! of scope and intentionally omitted.

use regex::Regex;

/// Truncate `prompt` to `generation_length` tokens and right-pad with zeros so
/// the result has exactly `generation_length` tokens.
///
/// Port of `get_padded_prompt`:
/// `single_prompt[:generation_length] + [0] * (generation_length - len(...))`
pub fn get_padded_prompt(prompt: &[i64], generation_length: usize) -> Vec<i64> {
    let mut padded: Vec<i64> = prompt.iter().take(generation_length).copied().collect();
    padded.resize(generation_length, 0);
    padded
}

/// Return the substring after the first occurrence of `" response"`, or the
/// whole input if `" response"` is not present.
///
/// Port of `strip_thoughts`.
pub fn strip_thoughts(txt: &str) -> &str {
    const NEEDLE: &str = " response";
    match txt.find(NEEDLE) {
        Some(i) => &txt[i + NEEDLE.len()..],
        None => txt,
    }
}

/// Build the ReasoningGym prompt for a given question.
///
/// Port of `make_rg_prompt`.
pub fn make_rg_prompt(question: &str) -> String {
    format!(
        "User: You are a helpful assistant. You first think about the reasoning \
         process in your mind and then provide the user with the answer. Question: \
         {question}. Assistant: <think"
    )
}

/// Extract the last number-ish token from `text` and clean it up.
///
/// Port of `extract_predicted_answer`. The Python logic:
///
/// ```python
/// regex_pattern = r"(-?[$0-9.,]{2,})|(-?[0-9]+)"
/// regexes_to_ignore = [",", r"\$", r"(?s).*#### ", r"\.$"]
/// match = re.findall(regex_pattern, text)
/// if match:
///     match = match[-1]                 # last match; a tuple of the 2 groups
///     match = [m for m in match if m][0] # first non-empty group
///     text = match.strip()
///     for rgx in regexes_to_ignore:
///         text = re.sub(rgx, "", text)
///     return text
/// else:
///     return None
/// ```
///
/// Because the alternation is exclusive, the whole-match text equals whichever
/// group participated, so `find_iter` reproduces the "first non-empty group"
/// selection. The four ignore replacements are applied in order with
/// `replace_all` (matching `re.sub(..., "")`).
pub fn extract_predicted_answer(text: &str) -> Option<String> {
    let pattern = Regex::new(r"(-?[$0-9.,]{2,})|(-?[0-9]+)").unwrap();

    let mut last: Option<String> = None;
    for m in pattern.find_iter(text) {
        last = Some(m.as_str().to_string());
    }
    let matched = last?;

    let mut cleaned = matched.trim().to_string();
    for rgx in [",", r"\$", r"(?s).*#### ", r"\.$"] {
        let re = Regex::new(rgx).unwrap();
        cleaned = re.replace_all(&cleaned, "").into_owned();
    }
    Some(cleaned)
}

/// Extract the ground truth as the trimmed text after the last `"####"`.
///
/// Port of `extract_ground_truth`: `text.split("####")[-1].strip()`.
pub fn extract_ground_truth(text: &str) -> String {
    text.split("####")
        .last()
        .unwrap_or("")
        .trim()
        .to_string()
}

/// Compare a generated answer against the provided solution's ground truth.
///
/// Port of `check_accuracy`. An unparseable model answer scores 0.0.
pub fn check_accuracy(generated_ans: &str, solution: &str) -> f64 {
    let ground_truth_answer = extract_ground_truth(solution);
    let model_answer = extract_predicted_answer(generated_ans.trim());
    match model_answer {
        Some(answer) if answer == ground_truth_answer => 1.0,
        _ => 0.0,
    }
}

/// Score a generated answer that follows a `" response"` marker by checking
/// only the text after that marker. Port of `single_fitness` (the unused extra
/// `i` argument in Python is dropped).
pub fn single_fitness(generated_answer: &str, true_answer: &str) -> f64 {
    const NEEDLE: &str = " response";
    match generated_answer.find(NEEDLE) {
        Some(find_idx) => {
            let true_idx = find_idx + NEEDLE.len();
            check_accuracy(&generated_answer[true_idx..], true_answer)
        }
        None => 0.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strip_thoughts_removes_thought_prefix() {
        assert_eq!(strip_thoughts("hello response42"), "42");
    }

    #[test]
    fn strip_thoughts_unchanged_without_marker() {
        assert_eq!(strip_thoughts("no marker here"), "no marker here");
    }

    #[test]
    fn strip_thoughts_first_occurrence_suffix() {
        assert_eq!(strip_thoughts("a responseB responseC"), "B responseC");
    }

    #[test]
    fn get_padded_prompt_pads_short_prompt() {
        assert_eq!(get_padded_prompt(&[1, 2, 3], 5), vec![1, 2, 3, 0, 0]);
    }

    #[test]
    fn get_padded_prompt_truncates_long_prompt() {
        assert_eq!(get_padded_prompt(&[1, 2, 3, 4, 5], 3), vec![1, 2, 3]);
    }

    #[test]
    fn make_rg_prompt_matches_expected_format() {
        assert_eq!(
            make_rg_prompt("2+2"),
            "User: You are a helpful assistant. You first think about the reasoning \
             process in your mind and then provide the user with the answer. Question: \
             2+2. Assistant: <think"
        );
    }

    #[test]
    fn extract_predicted_answer_simple_number() {
        assert_eq!(extract_predicted_answer("The answer is 42"), Some("42".to_string()));
    }

    #[test]
    fn extract_predicted_answer_money_number() {
        // "foo $1,234.56" -> last match token "$1,234.56" -> drop "," and "$" -> "1234.56"
        assert_eq!(
            extract_predicted_answer("foo $1,234.56"),
            Some("1234.56".to_string())
        );
    }

    #[test]
    fn extract_predicted_answer_no_numbers() {
        assert_eq!(extract_predicted_answer("no numbers here"), None);
    }

    #[test]
    fn extract_predicted_answer_with_ground_truth_marker() {
        // The `(?s).*#### ` replacement (a no-op on the already-token-reduced
        // string, since matched tokens cannot contain '#') still runs; the
        // final answer is the number after "#### ".
        assert_eq!(extract_predicted_answer("Question #### 42"), Some("42".to_string()));
        assert_eq!(extract_predicted_answer("#### 1234"), Some("1234".to_string()));
    }

    #[test]
    fn extract_ground_truth_splits_then_trims() {
        assert_eq!(extract_ground_truth("question #### 42"), "42");
        assert_eq!(extract_ground_truth("  no marker  "), "no marker");
    }

    #[test]
    fn check_accuracy_correct_wrong_and_unparseable() {
        // correct
        assert_eq!(check_accuracy("42", "question #### 42"), 1.0);
        // wrong
        assert_eq!(check_accuracy("43", "question #### 42"), 0.0);
        // unparseable -> None -> 0.0
        assert_eq!(check_accuracy("hello world", "question #### 42"), 0.0);
    }

    #[test]
    fn single_fitness_delegates_after_marker() {
        // Text after " response" is "42 ### 42", which extracts "42"; truth "42".
        assert_eq!(single_fitness("think response42", "#### 42"), 1.0);
        assert_eq!(single_fitness("think response43", "#### 42"), 0.0);
    }

    #[test]
    fn single_fitness_zero_without_marker() {
        assert_eq!(single_fitness("no response marker", "#### 42"), 0.0);
    }
}
