"""Descriptive uncertainty and case diagnostics from frozen predictions only."""
import argparse,json,math,hashlib
from pathlib import Path
import numpy as np


def wilson(k,n):
    z=1.959963984540054;p=k/n;den=1+z*z/n
    centre=(p+z*z/(2*n))/den
    radius=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [centre-radius,centre+radius]


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():p.error('Choose a new output file')
    source=Path('evidence/final_evaluation/run_01/predictions.json')
    rows=json.loads(source.read_text());result=[];rng=np.random.default_rng(42)
    for collection,model,method in [('single_digits','cnn','components'),('numbers','cnn','components'),
                                    ('single_letters','resnet18','grouped_components'),('strings','resnet18','grouped_components')]:
        group=[r for r in rows if (r['collection'],r['model'],r['method'])==(collection,model,method)]
        assert len(group) in (20,40)
        writers=sorted(set(r['writer'] for r in group))
        scores=np.array([np.mean([r['correct'] for r in group if r['writer']==w]) for w in writers])
        bootstrap=scores[rng.integers(0,len(writers),size=(10000,len(writers)))].mean(axis=1)
        correct=sum(r['correct'] for r in group)
        result.append(dict(collection=collection,model=model,n=len(group),correct=correct,
            case_sensitive_accuracy=correct/len(group),wilson95_image_iid=wilson(correct,len(group)),
            descriptive_writer_bootstrap95=np.quantile(bootstrap,[.025,.975]).tolist(),
            writer_accuracy=dict(zip(writers,scores.tolist())),
            case_insensitive_correct=sum(r['reference'].lower()==r['prediction'].lower() for r in group),
            case_only_errors=sum(r['reference']!=r['prediction'] and r['reference'].lower()==r['prediction'].lower() for r in group)))
    value=dict(source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),results=result,
        interpretation='Case-sensitive accuracy remains primary. Case-insensitive counts are diagnostic only. Wilson intervals assume independent photos, violated by repeated writers. Writer bootstrap is exploratory with only four known writers and does not establish unseen-writer generalisation. No training or inference performed.',
        bootstrap=dict(seed=42,repeats=10000,unit='writer',equal_samples_per_writer=True))
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value,indent=2))


if __name__=='__main__':main()
