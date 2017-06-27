import pandas as pd


def export(result):
    output = pd.DataFrame(data={"sentiment": result})
    output.to_csv("Bag_of_Words_model.csv", index=False, quoting=3)