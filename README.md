# Mitsuba - TSVBRDF
A modified version of the <a href="https://github.com/mitsuba-renderer/mitsuba">Mitsuba</a> renderer customized for the Spatio-Temporal BRDFs project.

## Usage
The projects is implemented as a BSDF plugin that can be used like other BSDF plugins.
You have to se the path to the time-varying material and time of the a frame to be rendered (time in [0-1]):
```
<bsdf type="tsvbrdf" id ="Material">
    <string name="filepath" value="$material"/>
    <float name="time" value="$time"/>
</bsdf>
```

To generate the whole sequence, you can use a python script:
```
data/scripts/tsvbrdf/tsvbrdf.py
```

There is a <a href="https://benedikt-bitterli.me/resources/">teapot</a> with <a href="https://www.cs.columbia.edu/CAVE/databases/staf/staf.php">steel rusting</a>.
To render the scene you can run a bash script (from the same directory):
```
data/scenes/teapot/scene.sh
```
Note that you might need to adjust path and other in the script.

## Citation
If you use this code, please cite:
```
@Article{Meister2021,
	author = {Daniel Meister and Adam Posp\'{\i}\v{s}il and Imari Sato and Ji\v{r}\'{\i} Bittner},
	title = {{Spatio-Temporal BRDF: Modeling and Synthesis}},
	journal = {Computers and Graphics},
  volume = {97},
  pages = {279-291},
	year = {2021},
}
```
