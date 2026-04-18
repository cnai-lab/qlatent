# Questionnaire Utils

The **Questionnaire Utils** library provides tools to create and manage various types of questionnaires. You can create questionnaire objects of the following types:

- **QMNLI**
- **QMLM**
- **QNSP**
- **QCOLA**
- **Regular Questionnaire** (not bound to a specific translation method)

## Creating a Questionnaire Object

### Initializing an Empty Questionnaire

You can initialize an empty questionnaire object by providing the required parameters. Here are two examples:

```python
# Initializing an empty QMNLI questionnaire
ASI = QMNLIQuestionnaire(
    name="ASI",
    num_of_questions=22,
    factors=["B1", "BI", "BG", "BP", "H"],
    factor_grouping={"B": {"B1", "BI", "BG", "BP"}},
    full_name="Ambivalent Sexism Inventory"
)

# Initializing an empty regular questionnaire
ASI = Questionnaire(
    name="ASI",
    num_of_questions=22,
    factors=["B1", "BI", "BG", "BP", "H"],
    factor_grouping={"B": {"B1", "BI", "BG", "BP"}},
    full_name="Ambivalent Sexism Inventory"
)
```

#### Parameters

- **`name`** (*str*): The name of the questionnaire.
- **`full_name`** (*str*, optional): The full name of the questionnaire.
- **`num_of_questions`** (*int*): The total number of questions the questionnaire will contain upon completion.
- **`factors`** (*list[str]*): A list of all unique factors for the questionnaire.
- **`factor_grouping`** (*dict[str, set[str]]*, optional): Allows grouping multiple factors into a single new factor for Cronbach's alpha calculations and factor correlations.

---

### Initializing a Questionnaire from a List of Question Classes

Alternatively, you can create a questionnaire object using a pre-defined list of question classes. Here's an example:

```python
# Imagine a list of question classes defined as follows
big5_qmlm_list = [
    BIG5Q1, BIG5Q2, BIG5Q3, BIG5Q4, BIG5Q5,
    BIG5Q6, BIG5Q7, BIG5Q8, BIG5Q9, BIG5Q10,
    BIG5Q11, BIG5Q12, BIG5Q13, BIG5Q14
]

# Creating a QMLM questionnaire from a list of questions
big5_qmlm = QMLMQuestionnaire.create_questionnaire_from_questions(big5_qmlm_list)

# Creating a regular questionnaire from a list of questions
big5 = Questionnaire.create_questionnaire_from_questions(big5_qmlm_list)
```

---

## Updating a Questionnaire Object

To update a questionnaire object, you can use the following methods:

### `set_factor_grouping`

Sets the factor grouping after the questionnaire has already been initialized.

#### Example:

```python
# big5 is a questionnaire that has already been initialized
# and has the following factors: 'Openness to Experience', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism'
big5.set_factor_grouping({
    "group1": {"Agreeableness", "Neuroticism"},
    "group2": {"Openness to Experience", "Conscientiousness"}
})
```

This code will set the factor grouping and create two groups, named `group1` and `group2`. Each group contains some of the current factors in the questionnaire.

### `add_question`

Adds a question object to the questionnaire.

#### Example:

```python
q = ASIQ1()  # An object of a question
ASI.add_question(q)
```

### `remove_question_by_ordinal`

Removes a question object from the questionnaire by specifying the question's ordinal.

#### Example:

```python
ASI.remove_question_by_ordinal(1)
```

### `remove_question_by_object`

Removes a question object from the questionnaire by specifying the question object itself. Note that the question object must be the exact same instance residing in the questionnaire (i.e., same memory address).

#### Example:

```python
q = ASIQ1()  # An object of a question
ASI.add_question(q)

# ... other operations ...

ASI.remove_question_by_object(q)
```

---

## Running the Questionnaire

To run models on the questionnaire, you need to call the following method in the order shown:

### `run`

This method runs models on the questionnaire questions with the selected configurations.

#### Parameters:

- **`pipelines`** (*list[str]*): The pipelines you wish to run on the questionnaire.
- **`questions_ordinals`** (*list[int]*, optional, defaults to all questions): A list specifying which questions to run.
- **`result_path`** (*Path*, optional, defaults to `./results/result.csv`): The path to save the results CSV.
- **`softmax`** (*list[str]*, optional, defaults to no softmax): Specifies which dimensions to apply softmax to.
- **`filters`** (*dict[str, dict]*, optional, defaults to no filters): A dictionary specifying filters to apply to the questions. Each filter is a key-value pair where the key is the filter name and the value is a filter function, defined as `'filtername': filter_function(q)`.

#### Example for a Questionnaire of Type `QMNLI`:

```python
pipelines = [
    pipeline(task="zero-shot-classification", device=0, model="MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7", trust_remote_code=True),
    pipeline(task="zero-shot-classification", device=0, model="facebook/bart-large-mnli", trust_remote_code=True),
]

big5_qmnli.run(
    pipelines=pipelines,
    softmax=['index', 'frequency'],  # Dimensions to apply softmax to in this example
    filters={
        "unfiltered": lambda q: {},
        "positive_only": lambda q: q.get_filter_for_postive_keywords(),
        "light_frequency": lambda q: get_filter_for_light_frequency_weights(q)
    },
    result_path="./results/big5_qmnli.csv"
)
```

After executing the `run` method, a CSV summarizing all the run data will appear in the specified `result_path`. This CSV can be used to generate summary data for the questionnaire.

> **Note:** The `run` method only appends rows to an existing CSV and creates an empty CSV if it is not found.

---

### `calc_correlations`

Calculates the correlation table for all factors, including new factors defined in `factor_grouping`.

#### Parameters:

- **`run_path`** (*str*): The path of the CSV file created by the `run` function to load results of models that were run on the questionnaire.
- **`correlations_path`** (*str*): The path to save the correlation table (in CSV format).

#### Example:

```python
big5_qmlm.calc_correlations(run_path="./results/big5_qmlm.csv", correlations_path="./results/big5_qmlm_correlations.csv")
```

---

### `report`

Generates a summary report for the questionnaire based on the models that were run.

#### Parameters:

- **`run_path`** (*str*): The path of the CSV file created by the `run` function to load results of models that were run on the questionnaire.
- **`output_path`** (*str*): The path to save the generated report (in Markdown `.md` format).
- **`template_path`** (*str*, optional, default: `"./report_template.md"`): The path to the report's template file, which defines the format of the final report before injecting questionnaire run data.

#### Example:

```python
big5_qmlm.report(
    run_path="./results/big5_qmlm.csv",
    output_path="./results/big5_qmlm_report.md",
    template_path="./report_template.md"
)
```

