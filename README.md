# Log Space Exploration: Color Constancy
Repository containing exploring effectiveness of log chromaticty space in color constancy task

## Data
### [SimpleCube++](https://github.com/Visillect/CubePlusPlus/blob/master/description/description.md#simplecube)

- quantity: $2234$ images
- bit: $16$-bit
- size: $648 \times 432$
- chromaticity space: 
- format: PNG
- black level: $\approx 2048$
- saturation level: $\le16384$
- illumination:  $(r,g,b) \quad \textit{s.t.} \quad r+g+b = 1$
- other
    - The right bottom rectangle $175 \times 250$ is cropped out to remove SpyderCube color target

### [Shi's Re-processing of Gehler's Raw Dataset](https://www2.cs.sfu.ca/~colour/data/shi_gehler/#DATA)
- quantity: $568$ images
- bit: $12$-bits
- size: $2041 \times 1359$ (for Canon 1D) for $2193 \times 1460$ (for Canon 5D)
- chromaticity space: camera RGB space
- format: PNG
- black level: $0$ (for Canon 1D) and $129$ (for Canon 5D)
- saturation level:
- illumination:  RGB in $12$-bit

## Experiment
### CCCNN
#### notice
- It takes $32\times 32$ random patches from images, implementation here might be diffent from one in the paper.

#### GehlerShi
##### notice
- Cannot access [L. Shi and B. V. Funt. Re-processed version of the gehler color constancy database of 568 images](http://www.cs.sfu.ca/colour/data) which claimed to be $14$ bits, thus used [Shi's Re-processing of Gehler's Raw Dataset](https://www2.cs.sfu.ca/~colour/data/shi_gehler/#DATA) which is actually $12$ bits.


#### SimpleCube++
- (maximum $212$ patches)

|        |  Log7 | Log5   | Linear5 |
| ------ | ------ | ------ | ------ |
|  *min*   | **0.028** | 0.539  | 0.086  |
|  *10%*   | **0.379** | 0.932  | 0.747  |
|  *med*   | **1.114** | 1.538  | 1.866  |
|  *avg*   | **4.432** | 4.793  | 4.861  |
|  *90%*   | 12.598 | 12.585 | **12.427** |
|  *max*   | **34.210** | 34.339| 34.738 |
