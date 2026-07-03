
import pandas as pd
import matplotlib.pyplot as plt

def summarize(csv,outdir):
    df=pd.read_csv(csv)
    s=df.groupby(["artifact","level"])["decoded"].mean().reset_index()
    s.to_csv(outdir/"summary.csv",index=False)
    for art,g in s.groupby("artifact"):
        plt.figure(figsize=(5,3))
        plt.plot(g.level,g.decoded,marker="o")
        plt.xlabel("Level")
        plt.ylabel("Decode rate")
        plt.title(art)
        plt.grid(True)
        plt.savefig(outdir/f"{art}.png",dpi=150)
        plt.close()
    rows=[]
    for art,g in s.groupby("artifact"):
        below=g[g.decoded<=0.5]
        infl=below.level.iloc[0] if len(below) else None
        rows.append((art,infl))
    print("\nApproximate 50% decode thresholds")
    for r in rows:
        print(r)
