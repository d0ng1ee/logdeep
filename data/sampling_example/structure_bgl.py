# pipeline
# 1. read origin logs
# 2. extract label, time and origin event
# 3. match event to the template
import re
import pandas as pd
from tqdm import tqdm
para = {"bgl":"bgl/bgl2_100k","template":"bgl/templates.csv","structured_file":"bgl/BGL_100k_structured.csv"}
# read origin logs
def data_read(filepath):
    fp = open(filepath, "r")
    datas = []
    lines = fp.readlines()
    i = 0
    for line in tqdm(lines):
        row = line.strip("\n").split()
        datas.append(row)
        i = i + 1
    fp.close()
    return datas

def match(BGL):
    # match event to the template
    template = pd.read_csv(para["template"])

    event = []
    event2id = {}

    for i in range(template.shape[0]):
        event_id = template.iloc[i, template.columns.get_loc("EventId")]
        event_template = template.iloc[i, template.columns.get_loc("EventTemplate")]
        event2id[event_template] = event_id
        event.append(event_template)

    error_log = []
    eventmap = []
    print("Matching...")
    for log in tqdm(BGL):
        log_event = " ".join(log[9:])
        for i,item in enumerate(event):
            if re.match(r''+item,log_event) and re.match(r''+item,log_event).span()[1] == len(log_event):
                eventmap.append(event2id[item])
                break
            if i == len(event)-1:
                eventmap.append('error')
                error_log.append(log_event)
    return eventmap
def structure(BGL,eventmap):
    # extract label, time and origin event
    label = []
    time = []
    for log in tqdm(BGL):
        label.append(log[0])
        time.append(log[4])

    BGL_structured = pd.DataFrame(columns=["label","time","event_id"])
    BGL_structured["label"] = label
    BGL_structured["time"] = time
    BGL_structured["event_id"] = eventmap
    # Remove logs which do not match the template(very few logs ......)
    BGL_structured = BGL_structured[(-BGL_structured["event_id"].isin(["error"]))]
    BGL_structured.to_csv(para["structured_file"],index=None)

if __name__ == "__main__":
    BGL = data_read(para["bgl"])
    eventmap = match(BGL)
    structure(BGL,eventmap)
