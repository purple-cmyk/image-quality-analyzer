
from pathlib import Path
import random
import pandas as pd
import cv2
from tqdm import tqdm

import degradations as dg
from decoder import decode
from analysis import summarize

root=Path(__file__).parent
inp=root/"qr_codes"
out=root/"output"
rows=[]

levels={
"blur":[0.5,1,2,3,4,5],
"motion":[3,5,9,15,21],
"noise":[2,5,10,20,30,40],
"exposure":[0.3,0.5,0.7,1.3,1.6,2.0],
"contrast":[0.2,0.5,0.8,1.5,2.0],
"haze":[0.1,0.2,0.4,0.6,0.8],
"color":[0.2,0.5,1.0,1.5,2.0]
}

imgs=list(inp.glob("*"))

for p in tqdm(imgs):
    img=cv2.imread(str(p))
    if img is None:
        continue

    for l in levels["blur"]:
        ok=decode(dg.blur(img,l))
        rows.append([p.name,"blur",l,ok])

    for l in levels["motion"]:
        ang=random.uniform(0,180)
        ok=decode(dg.motion_blur(img,l,ang))
        rows.append([p.name,"motion",l,ok])

    for l in levels["noise"]:
        ok=decode(dg.gaussian_noise(img,l))
        rows.append([p.name,"noise",l,ok])

    for l in levels["exposure"]:
        ok=decode(dg.exposure(img,l))
        rows.append([p.name,"exposure",l,ok])

    for l in levels["contrast"]:
        ok=decode(dg.contrast(img,l))
        rows.append([p.name,"contrast",l,ok])

    for l in levels["haze"]:
        ok=decode(dg.haze(img,l))
        rows.append([p.name,"haze",l,ok])

    for l in levels["color"]:
        ok=decode(dg.colorfulness(img,l))
        rows.append([p.name,"color",l,ok])

df=pd.DataFrame(rows,columns=["image","artifact","level","decoded"])
csv=out/"results.csv"
df.to_csv(csv,index=False)
summarize(csv,out)
print("Finished.")
