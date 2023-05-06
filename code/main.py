import run
import argparse

# pass parameters to the script
pa = argparse.ArgumentParser(description='manual to this script')
pa.add_argument('--ty_n', type=int, default = None)
ar = pa.parse_args()

run.run_model(ty_n = ar.ty_n)











