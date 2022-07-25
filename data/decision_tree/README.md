Normally, I would explain what is what through documentation in the code. But we have JSON files so I'm doing the explanations all here.

In case you're lost, you are at the decision_tree config folder. The JSON files here are used to create the pre-processing that is needed to create a DataFrame object to feed to your decision tree.

Each JSON file is comprised of three keys:
- `DataFrameColumns`: Each column in the dataframe belongs to one the three types: `Binary`, `Number`, `Keyword`. The Binary columns are the ones that their answers are either yes or no. We use these columns to mark the different situations.<br>
The Number columns are the ones that their answers are numbers.<br>
The Keyword columns are the ones that their answers require looking at the user's answer and selecting a value from it.
- `KeyWords`: This is the list keywords that is cross-checked with the `Keyword` columns above. You see, each keyword can have many synonyms. Zum bei spiel, we consider all ["absent", "unjustified", "unauthorized", "absence", "unjustified absence", "unauthorized absence", "unjustified or unauthorized absence "] as referring to the absent case. So if we see them user's response, we know that the user is talking about the absent case.
- `Mapping`: So this where it gets messy. Since a version of this decision tree is in use in their production code, I was not allowed to create a completly new paradigm for the decision tree else I would break their produciton code. So I decided to create a mapping between the columns and the keywords. The only way that you can understand what this is, is by tracing it. Sorry UwU.
