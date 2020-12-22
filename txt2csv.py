# coding: utf-8
import csv

headers = ["video_id","question","answer","pos_tag","video_start_time","video_end_time"]  # title

txtfiles = ["train.txt","val1.txt","val2.txt"]
csvfiles = ["train.csv","val1.csv","val2.csv"]
for fid in range(3):
    with open(txtfiles[fid],"r",encoding = "utf-8") as txtf:
        with open(csvfiles[fid],'w',newline='',encoding = "utf-8")as csvf:
            csv_writer = csv.writer(csvf)
            csv_writer.writerow(headers)
            lines = txtf.readlines()
            for ind in range(0,len(lines),6):
                meta = [line[:-1] for line in lines[ind:ind+4]] # "video_id","question","answer","pos_tag"
                meta[1] = meta[1].replace("'","\\'")
                meta.append(eval(lines[ind+4])[0])              # "video_start_time"
                meta.append(eval(lines[ind+4])[1])              # "video_end_time"
                if(len(meta) == 6):
                    csv_writer.writerow(meta)
