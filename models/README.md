This folder is used as an archive for storing past models in case you need to revert to a previous version.<br>
Ignore the `nlg` folder since we are now using pre-trained BERT models for this purpose.
The `nlu` and `past_models_archive` folders are used to store the models that are were trained and used previously.<br>
Personally, I use the `past_models_archive` as a dump of previous that were trained alongside the final model which was used and everything in the `nlu` folder is separated into categories corresponding to the different models that were in use before the new version was deployed.
For example, the `lackodrespect_customer` folder in the the `nlu` directory contains the models which were in use before deploying the new model for lack_of_respect.