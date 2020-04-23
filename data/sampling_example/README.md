# Example of how to sample your own log

*Select first 100k logs from bgl and hdfs dataset for demo.*

## BGL

The bgl dataset contains only time information, so it is suitable for time windows

### 1. Structure your own log

- read origin logs
- extract label, time and origin event
- match event to the template id
  
*"-" label in bgl represent normal, else label is abnormal.*

`python structure_bgl.py`

### 2. Sampling with sliding window or fixed window

Use the time window for sampling by calculating the time difference between different logs.

The unit of window_size and step_size is hour.

If `step_size=0`, it used fixed window; else, it used sliding window

`python sample_bgl.py`

## HDFS

The bgl dataset contains block_id information, so it is suitable for grouping by block_id

*block_id represents a designated hard disk storage space*

### 1. Structure your own log

same as bgl...

### 2. Sampling with block_id

`python sample_hdfs`