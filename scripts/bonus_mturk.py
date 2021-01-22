# coding: utf-8
import boto3
import json

MTURK_SANDBOX = 'https://mturk-requester-sandbox.us-east-1.amazonaws.com'

client = boto3.client('mturk',
   aws_access_key_id = "PASTE_YOUR_IAM_USER_ACCESS_KEY",
   aws_secret_access_key = "PASTE_YOUR_IAM_USER_SECRET_KEY",
   region_name='us-east-1', # always ‘us-east-1’ for MTurk.
#    endpoint_url = MTURK_SANDBOX # connect to sandbox
)

if __name__ == "__main__":
    with open("bonus_info.json","r") as f:
        bonus_info = json.load(f)

    worker_amount = len(bonus_info["WorkerIds"])
    bonus_amount = 0
    for i in range(worker_amount):
        try:
            response = client.send_bonus(
                WorkerId=bonus_info["WorkerIds"][i],
                BonusAmount=bonus_info["BonusAmounts"][i],
                AssignmentId=bonus_info["AssignmentIds"][i],
                Reason=bonus_info["Reasons"][i],
                UniqueRequestToken=bonus_info["UniqueRequestTokens"][i]  #  useful in cases such as network timeouts
            )
            bonus_amount += 1
        except Exception as e:
            print(f"WorkerId: {bonus_info['WorkerIds'][i]},\nAssignmentId: {bonus_info['AssignmentIds'][i]},\n"+
                f"UniqueRequestToke: {bonus_info['UniqueRequestTokens'][i]}\n{e}\n")
    
    print(f"Successfully bonused {bonus_amount}/{worker_amount}")
