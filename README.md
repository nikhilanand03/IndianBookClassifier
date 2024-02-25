# IndianBookClassifier
This is a classifier that tells us if a given book has an Indian context, i.e, it is based on Indian themes and ideas or by an Indian author.

## Model pipeline

![Prompt Template (2)](https://github.com/nikhilanand03/IndianBookClassifier/assets/75153414/dc0c994f-e395-4ad9-8826-1c981e12eb80)

My pipeline predicts whether a book Indian without the need for inputting the label. It uses a zero-shot method for labelling, and then a smaller model is trained on the labelled dataset. The Mixtral LLM with 7B parameters is used to label my dataset, and the labelled dataset is then used to train a much smaller model with 80M parameters. Thus, we have a very efficient model with few parameters, that could work with the level of efficiency of a 7B-parameter model. It is finetuned specifically for the task of predicting whether a book has an Indian context or not, ie, it is based on Indian themes and ideas or by an Indian author.

## Dataset

All of the data was obtained from the Libgen database (https://data.library.bz/dbdumps/). To replicate this project, it can be downloaded from the "fiction.rar" link on the page.

## Project Structure

1. I started with collecting the data and cleaning it. You can find the code for this step, along with the labelling pipeline, in the /data/ directory. Once the fiction.sql file was extracted from the fiction.rar file into this directory, I used the "extract_data.ipynb" code file to read the sql commands and create a csv out of it. There were 3 million books and their data present here.
2. Next, the 3 million books were compiled into the file "csv_full_fiction.csv" through running "compile_csvs.ipynb".
3. The dataset had duplicates, so these were removed in "clean_dataset.ipynb".
4. In the /llms/ directory, "llm_labelling_script.ipynb" has the labelling code. Replicate's API was used to run calls to Mixtral-8x7B and the final data was stored in files.
5. "txt2data_conv.ipynb" converted the text data to a csv file and finally in "/data/merging dataset" the data was merged. You can find the merged dataset here.
6. The /roberta model/ directory contains the training loop of the RoBERTa model in roberta_model.ipynb.

### Challenges

Due to lack of GPUs and replicate credits, I could only obtain labels for 640 out of 3 million data points. Still, the performance was quite good on the given dataset, as we'll see in the next section. I class-balanced the final merged dataset to contain 200 Indian and 200 Non-Indian books.

I believe that once more data and features are provided during training, the model will be able to read between the lines and answer more accurately for more challenging examples, where the author is not Indian but perhaps the Indian context is hidden in the summary or title.

## Training results

Training was carries out for a single epoch and the mean loss varied across batches as follows:

<img width="575" alt="loss_graph" src="https://github.com/nikhilanand03/IndianBookClassifier/assets/75153414/c007040c-1a1c-46d8-857d-f0182bfe9f37">

The loss can be seen to be decreasing monotonically. Furthermore, on unseen validation data coming from the same dataset, the model had an accuracy of **98%** with only 2 incorrect answers.

<img width="297" alt="Screenshot 2024-02-25 at 4 28 31 PM" src="https://github.com/nikhilanand03/IndianBookClassifier/assets/75153414/6d69ee0f-9065-4e14-856b-7c5e6526a5df">

## Testing

The model can be tested using custom input using the "model_prediction.py" file. Before running this file, download the model's pth file from [this](https://drive.google.com/file/d/1a8Qp8WsIvKlptIrUlgmhFbjqfMcTpXap/view?usp=sharing) link. You can input any title name, author name, series name, and language and receive a response from the model.

<img width="267" alt="Screenshot 2024-02-25 at 4 30 51 PM" src="https://github.com/nikhilanand03/IndianBookClassifier/assets/75153414/c461015c-b278-4c3a-b103-97023a6994a6">


