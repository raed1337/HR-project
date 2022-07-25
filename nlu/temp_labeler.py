import pandas as pd

job_tiles = pd.read_csv("Job Titles.csv", header=None)
seq_in = open("seq_in.csv", "w+")
seq_out = open("seq_out.csv", "w+")
label = open("label.csv", "w+")
for row in job_tiles.iterrows():
    words = list(row[1])[0].split()
    seq_in_word = " ".join(words).lower()
    seq_out_words = " ".join(
        ["B-position" if word_index == 0 else "I-position" for word_index, word in enumerate(words)]
    )
    label_word = "informworkposition"
    seq_in.write(seq_in_word + "\n")
    seq_out.write(seq_out_words + "\n")
    label.write(label_word + "\n")
seq_in.close()
seq_out.close()
label.close()
