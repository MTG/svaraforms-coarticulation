
# Svara-forms and Coarticulation in Carnatic Music: An Investigation Using Deep Clustering

This repository accompanies the following paper [[pdf](https://repositori.upf.edu/handle/10230/60674)]:
```Thomas Nuttall, Xavier Serra, and Lara Pearson. "Svara-forms and Coarticulation in Carnatic Music: An Investigation Using Deep Clustering". The 11th International Conference on Digital Libraries for Musicology, DLFM 2024```


## Data Requirements
This article uses a performance from the Saraga Dataset of the composition Kamakshi by Carnatic vocalist, Sanjay Subrahmanyan in raga Bhairavi [1]. We present as part of this work, note-level annotations for this performance - [data/annotations/kamakshi.txt](data/annotations/kamakshi.txt)

## Repository Structure

### Data
All svara pitch plots, audios and static features can be found in `results/svara_plots`

### Code and Scripts
The scripts to reproduce the paper results can be found in `experiments` 

### Paper Results
The clustering results can be found in `results/clustering`. The application code for the svara-form explorer is found in `app/` . The visualisation can be found online here - [https://svara-clustering.onrender.com/](https://svara-clustering.onrender.com/). The hosting is provided using the free onrender servers and as such if the site has not been visited for a long time, it may take several minutes to load.

### SHAP Analysis
The SHAP analysis results can be found in `results/SHAP`. Further details are provided in that folder.

## License
This repository is released under the terms of the GNU Affero General Public License (v3 or later). See the COPYING file for more information. 

## Citing
This work can be cited with the following Bibtex...

```
@inproceedings{nuttall2024svara,
  title={Svara-forms and coarticulation in Carnatic music: an investigation using deep clustering},
  author={Nuttall, Thomas and Serra, Xavier and Pearson, Lara},
  booktitle={Proceedings of the 11th International Conference on Digital Libraries for Musicology},
  pages={15--22},
  year={2024}
}
```

## References

[1] Srinivasamurthy, A., Gulati, S., Repetto, R. C., & Serra, X. (2021). Saraga: open datasets for research on indian art music. _Empirical Musicology Review_, _16_(1), 85-98.

[2] CompIAM repository of datasets, tools, and models for the computational analysis of Carnatic and Hindustani music - https://github.com/MTG/compIAM

[3] Plaja-Roglans, G., Nuttall, T., Pearson, L., Serra, X., & Miron, M. (2023). Repertoire-specific vocal pitch data generation for improved melodic analysis of carnatic music. _Transactions of the International Society for Music Information Retrieval_, _6_(1), 13-26.

[4] Maghoumi, Mehran, and Joseph J. LaViola. "DeepGRU: Deep gesture recognition utility." _Advances in Visual Computing: 14th International Symposium on Visual Computing, ISVC 2019, Lake Tahoe, NV, USA, October 7–9, 2019, Proceedings, Part I 14_. Springer International Publishing, 2019.