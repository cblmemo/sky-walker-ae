python3 fig_2.py
python3 fig_3_a.py
python3 fig_3_b.py
python3 fig_4_a.py
python3 fig_4_b.py
# Notice: data for Figure 5(a) cames from the script fig_5_b.py.
python3 fig_5_a.py
# Notice: Figure 5(b) originally uses --sample-size 10000, but that requires substantial compute resources. Using 1000 here for ease of reproducibility.
python3 fig_5_b.py --sample-size 1000 --user-field state
python3 fig_6.py

if [ ! -d metric ]; then
    echo "Extracting metric.tar.gz..."
    tar -xzvf metric.tar.gz
else
    echo "metric directory already exists."
fi
python3 eval_plots.py
