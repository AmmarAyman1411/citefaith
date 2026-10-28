from datasets import load_dataset
configs = ['covidqa','cuad','delucionqa','emanual','expertqa','finqa','hagrid','hotpotqa','msmarco','pubmedqa','tatqa','techqa']
for c in configs:
    d = load_dataset('rungalileo/ragbench', c)
    d.save_to_disk(f'data/ragbench/{c}')