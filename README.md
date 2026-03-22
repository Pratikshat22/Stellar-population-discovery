# Stellar-population-discovery
Challenging ideal stellar models using real data: uncovering hidden populations, weak correlations, and non-ideal physical behavior
# Stellar Population Discovery Using Machine Learning

I worked on this project to see if I could use machine learning to find patterns in stellar data that might not be obvious just by looking at plots. The dataset has information about stars - their temperatures and luminosities - and I wanted to check whether the well-known Stefan-Boltzmann law (L ∝ T⁴) actually holds when you look at real data.

The idea was simple: if the law holds perfectly, temperature and luminosity should follow a tight relationship. But when I ran the numbers, I got something different.

# What I Found

The global scaling exponent came out to about 4.58, which is close to the theoretical 4.0 but not exactly. More interesting was the correlation - only 0.43. That means temperature alone doesn't explain luminosity very well. Something else is going on.

When I ran clustering (K-Means), the algorithm split the stars into three distinct groups. I didn't tell it how many groups to expect - it found them on its own.

| Group | Stars | Exponent | Notes |
|-------|-------|----------|-------|
| Group 0 | 72 | 2.17 | Some temperature dependence |
| Group 1 | 132 | 0.10 | Very weak dependence |
| Group 2 | 36 | 0.10 | Almost no dependence |

These look like different kinds of stars behaving differently. Group 1 and 2 have exponents close to zero - their luminosity doesn't really change with temperature. That's not what textbooks say should happen for most stars.

## Anomalies

The anomaly detection (Isolation Forest) flagged 12 stars as outliers. Some of these are extreme:

- One star at 3000K with 280,000 times the Sun's luminosity
- Several hot stars (20,000-25,000K) that are incredibly faint
- A 40,000K star that's 800,000 times brighter than the Sun

These might be unusual stellar types - red supergiants, white dwarfs, maybe even a Wolf-Rayet candidate. Hard to tell from just temperature and luminosity, but they're definitely not following the same rules as the others.

## Why This Matters

The Stefan-Boltzmann law works well for idealized blackbodies, but real stars are more complicated. This analysis shows that:

1. Different stellar populations follow different scaling relations
2. Temperature alone isn't enough to predict luminosity
3. There are stars that don't fit into any neat category

If I had more data (radius, maybe metallicity), the models might perform better. But even with just two variables, the patterns are there.

# Interactive Dashboard

I made a Plotly dashboard with a few plots:

1.) HR diagram with clusters colored separately
2.) Anomaly highlighting
3.) Deviation analysis (where stars fall relative to the model)
4.) 3D view (temperature, luminosity, and a fake distance just for visualization)

Open `file:///C:/Users/nmee8/Downloads/INDEX.HTML` in any browser. The 3D plot rotates, hover works, etc.

# Files

1.) `file:///C:/Users/nmee8/Downloads/INDEX.HTML` - interactive dashboard
2.) `analysis_script.py` - the actual code
3.) `README.md` - this file


```bash
pip install kagglehub plotly scikit-learn scipy
python analysis_script.py
