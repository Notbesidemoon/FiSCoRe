# Data

The datasets are not redistributed in this repository. Prepare the following
files before training or evaluating on ParaRel:

- `pararel_train.json`: a JSON list of training examples.
- `pararel_test.json`: a JSON list of evaluation examples.

Each example must use this schema:

```json
{
  "question": "What is the capital of France?",
  "answer": "Paris"
}
```

The paper trains on 15,000 ParaRel examples after filtering examples whose
semantic entropy is below 1. Follow the original dataset licenses when
constructing and redistributing these files.
