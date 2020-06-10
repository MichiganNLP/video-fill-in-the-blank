import csv
from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def getVideoFeatures(key, startFrame, endFrame, videoFeatures, textLen):
        feature_np = videoFeatures[key]['c3d_features'][startFrame:endFrame+1]
        
        if feature_np.shape[0] > 200:
            feature = np.zeros((200, feature_np.shape[1]))
            for i in range(200):
                feature[i] = feature_np[round(i * (feature_np.shape[0]-1)/199)]
        else:
            feature = feature_np

        return torch.tensor(feature, dtype=torch.float)

with open('val1_50_mturk_appr_answers.csv') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        video_id, question, start_time, end_time, _, standard_answer, worker_answers = row
        extended_answers = set(standard_answer + worker_answers)
        masked_sentence = tokenizer.tokenize(question)
        mask_position = sentence.index('[MASK]')

        sequence_id = tokenizer.encode(quesiton)