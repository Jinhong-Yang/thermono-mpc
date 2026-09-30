"""Map the public archive into the input layout used for diagnostic execution."""
from pathlib import Path
import argparse,hashlib,json,shutil

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def stage(archive,output):
    archive=archive.resolve();output=output.resolve()
    manifest=archive/'RELEASE_MANIFEST.json'
    if not manifest.is_file():raise FileNotFoundError('Extract the versioned research archive first')
    inventory={r['path']:r for r in json.loads(manifest.read_text(encoding='utf-8'))['files']}
    records=[]
    for source,destination in [('results/control','control_test_v2'),('data/prediction-test','prediction_test_v2')]:
        folder=archive/source
        if not folder.is_dir():raise FileNotFoundError(folder)
        for path in sorted(folder.iterdir()):
            if not path.is_file():continue
            name=path.relative_to(archive).as_posix();expected=inventory[name]['sha256']
            if sha(path)!=expected:raise ValueError('Archive member hash mismatch: '+name)
            target=output/destination/path.name
            if target.exists() and sha(target)!=expected:raise FileExistsError('Refusing to replace differing data: '+str(target))
            target.parent.mkdir(parents=True,exist_ok=True)
            if not target.exists():shutil.copy2(path,target)
            if sha(target)!=expected:raise ValueError('Staged copy hash mismatch')
            records.append({'source':name,'target':target.relative_to(output).as_posix(),'sha256':expected})
    (output/'staging_receipt.json').write_text(json.dumps({'files':records,'count':len(records)},indent=2),encoding='utf-8')
    return len(records)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive-root',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();print(json.dumps({'verified_files_staged':stage(a.archive_root,a.output)}))
