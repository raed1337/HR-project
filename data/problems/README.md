Normally, I would explain what is what through documentation in the code. But we have JSON files so I'm doing the explanations all here.

In case you're lost, you are at the problems config folder. The JSON files here are used dictate the flow of conversation (which question to ask and what to do if certain answers are given by the user to some questions). This is the most important part of the chatbot and its customization if I might say so myself. And I can say so myself because ___I___ wrote it.

Each JSON file here is comprised of two main keys:
- `MainProblemCategory`: This is the main category of the problem. This is the category that the users will be asked to answer in the second question that we ask of them (The one where we ask them to confirm what their problem is).
- `SubCategories`: Each problem might have multiple subcategories. Exempli Gratia, the absence category has 5 subcategories.

Now let's go through the subcategory keys:
- First is the name that we define the sub category as. This is what we check with NLU's output in order to find out what the user is talking about.
- The is the `QuestionSet`. This is the list of questions that we should ask the user for this subcategory IN ORDER.
- The `FlowChange` dictionary, determines what to do if a certin question was asked. Based on user's response and looking at `action` key's value, we either add or remove quesitons to user's question set.

And that's it. Now I know that reading these will do nothing for you. Fortunatly for you, I have standards. You can just run the `dialogue.py` in the dialogue_system directory to play around and trace a single test case.
