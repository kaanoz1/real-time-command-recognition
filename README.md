# Project Purpose:

**The goal of this project is to develop a deep learning model capable of recognizing specific voice commands (“yes” or “no”) during an ordinary conversation. The model will identify these commands with a certain level of confidence and display the results in the console.**


## About dataset:

All dataset files are located in the /data folder. The dataset consists of three categories:
- All recordings of the word “yes” are stored in /data/yes_data/
- All recordings of the word “no” are stored in /data/no_data/.
- Any other sounds, including daily conversation fragments and background noise, fall under the "unknown" category. Since these recordings may be longer than one second, they must be split into 1-second segments before training. The model will then be trained to recognize these segments as "unknown".

## Licence

Ain't no copyright or licence. Use it like however you want.