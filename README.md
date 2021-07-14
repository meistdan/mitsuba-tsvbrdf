# Mitsuba - TSVBRDF
A modified version of the <a href="https://github.com/mitsuba-renderer/mitsuba">Mitsuba</a> renderer customized for the <a href="https://www.sciencedirect.com/science/article/abs/pii/S0097849321000431">Spatio-Temporal BRDFs</a> project. We also released <a href="https://github.com/meistdan/tsvbrdf">the source code of the synthesis in another repository</a>.

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

## Dependencies
We use <a href="https://opencv.org/">OpenCV</a> for the data manipulation besides Mitsuba dependencies.
Tested with Visual Studio 2015 (x64).

## License
Mitsuba is licenced under GNU General Public License v3.0. Our code is released into the public domain.

## Citation
If you use this code, please cite <a href="https://www.sciencedirect.com/science/article/abs/pii/S0097849321000431">the paper</a>:
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
