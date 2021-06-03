# HDF5データセットの規格

## 目次

- [データ型](#datatype)
  - [数値](#datatype-scalar)
  - [画像](#datatype-image)
  - [pose](#datatype-pose)
  - [translation](#datatype-translation)
  - [rotation](#datatype-rotation)
  - [points](#datatype-points)
  - [color](#datatype-color)
- [ツリー構造](#structure)

<div id="datatype"></div>

## データ型

<div id="datatype-scalar"></div>

### 数値

|TYPE       |dtype       |default-range                                 |概要|
|-----------|------------|----------------------------------------------|----|
|`"uint8"`  |`np.uint8`  |`(0, 255)`                                    |符号なし 8bit 整数|
|`"int8"`   |`np.int8`   |`(-128, 127)`                                 |符号あり 8bit 整数|
|`"int16"`  |`np.int16`  |`(-32768, 32767)`                             |符号あり 16bit 整数|
|`"int32"`  |`np.int32`  |`(-2147483648, 2147483647)`                   |符号あり 32bit 整数|
|`"int64"`  |`np.int64`  |`(-9223372036854775808, 9223372036854775807)` |符号あり 64bit 整数|
|`"float16"`|`np.float16`|`(-inf, inf)`                                 |16bit 浮動小数点|
|`"float32"`|`np.float32`|`(-inf, inf)`                                 |32bit 浮動小数点|
|`"float64"`|`np.float64`|`(-inf, inf)`                                 |64bit 浮動小数点|

(`np` = `numpy`)

- ATTRIBUTE
  - type：TYPE
  - stamp.sec：タイムスタンプ(整数部)
  - stamp.nsec：タイムスタンプ(小数部)

<div id="datatype-image"></div>

### 画像

|TYPE           |dtype                   |shape      |default-range|概要|
|---------------|------------------------|-----------|-------------|----|
|`"mono8"`      |`np.ndarray<np.uint8>`  |`(H, W)`   |`(0, 255)`   |1ch 8bit のグレースケール.|
|`"mono16"`     |`np.ndarray<np.uint16>` |`(H, W)`   |`(0, 65535)` |1ch 16bit のグレースケール.|
|`"bgr8"`       |`np.ndarray<np.uint8>`  |`(H, W, 3)`|`(0, 255)`   |3ch 8bit のBGR画像.|
|`"rgb8"`       |`np.ndarray<np.uint8>`  |`(H, W, 3)`|`(0, 255)`   |3ch 8bit のRGB画像.|
|`"bgra8"`      |`np.ndarray<np.uint8>`  |`(H, W, 4)`|`(0, 255)`   |4ch 8bit のBGRA画像.|
|`"rgba8"`      |`np.ndarray<np.uint8>`  |`(H, W, 4)`|`(0, 255)`   |4ch 8bit のRGBA画像.|
|`"depth"`      |`np.ndarray<np.float32>`|`(H, W)`   |`(0.0, inf)` |32bit浮動小数点型でメートル単位の深度を補完した深度マップ. 値が存在しない画素は，`inf`とする.|
|`"semantic2d"` |`np.ndarray<np.uint8>`  |`(H, W)`   |`(0, 255)`   |1ch 8bit の画像で, ラベルの番号を画素に格納したSemantic Segmentation用ラベル画像. ラベルの情報は別途データセット内に格納する.|

(`np` = `numpy`)

- ATTRIBUTE
  - type：TYPE
  - stamp.sec：タイムスタンプ(整数部)
  - stamp.nsec：タイムスタンプ(小数部)
  - frame_id：座標系

<div id="datatype-pose"></div>

### pose

自己位置の値. *translation*と*rotation*のデータを含む.

- *translation*
- *rotation*
- ATTRIBUTE
  - type: `"pose"`
  - stamp.sec：タイムスタンプ(整数部)
  - stamp.nsec：タイムスタンプ(小数部)
  - frame_id：座標系
  - child_frame_id：座標系(子)

<div id="datatype-translation"></div>

### translation

並進ベクトル

- dtype  
  `np.ndarray<np.float64>`
- shape  
  `(3,)`
- ATTRIBUTE
  - type：`"translation"`
  - array：`"x,y,z"`

<div id="datatype-rotation"></div>

### rotation

クォータニオン

- dtype  
  `np.ndarray<np.float64>`
- shape  
  `(4,)`
- ATTRIBUTE
  - type：`"quaternion"`
  - array：`"x,y,z,w"`

<div id="datatype-points"></div>

### **points**

三次元点群

- dtype  
  `np.ndarray<np.float32>`
- shape  
  `(N, 3)`
- ATTRIBUTE
  - type：`"points"`
  - frame_id：座標系
  - file_path：元データのファイルパス

<div id="datatype-intrinsic"></div>

### intrinsic

カメラ内部パラメータ

- `Fx (float64)`
- `Fy (float64)`
- `Cx (float64)`
- `Cy (float64)`
- `height (uint32)`
- `width (uint32)`

- ATTRIBUTE
  - type：`"intrinsic"`

<div id="datatype-color"></div>

### color

色

- dtype  
  `np.ndarray<np.uint8>`
- shape  
  `(3,)`
- ATTRIBUTE
  - type：`"color"`
  - array：`"b,g,r"`

<div id="structure"></div>

## ツリー構造

```
.
├── "header"
│　　│　データセットの主要情報を記述
│　　│  
│　　├── "keys"
│　　│　　　　データセットのキーの名前(str)をnp配列として記述
│　　│　　　　(例) ['rgb' 'pose' 'semantic2d' 'depth']
│　　│　　　　　dtype: np.ndarray<str>
│　　│
│　　└── "length"
│　　　　　　　データセットの数を記述
│　　　　　　　(例) 3625
│　　　　　　　　dtype: np.int64
│
├── "K"
│　　│　データセット中の画像のカメラ内部パラメータを記述
│　　│
│　　├── "[キー1]"
│　　│　　│　データセット中のキーに対応
│　　│　　│  
│　　│　　├── "Fx"
│　　│　　│　　　　焦点距離(x軸方向)[px]
│　　│　　│　　　　　dtype: np.float64
│　　│　　│
│　　│　　├── "Fy"
│　　│　　│　　　　焦点距離(y軸方向)[px]
│　　│　　│　　　　　dtype: np.float64
│　　│　　│
│　　│　　├── "Cx"
│　　│　　│　　　　主点の位置(x軸方向)[px]
│　　│　　│　　　　　dtype: np.float64
│　　│　　│
│　　│　　├── "Cy"
│　　│　　│　　　　主点の位置(y軸方向)[px]
│　　│　　│　　　　　dtype: np.float64
│　　│　　│
│　　│　　├── "height"
│　　│　　│　　　　画像の高さ[px]
│　　│　　│　　　　　dtype: np.uint32
│　　│　　│
│　　│　　└── "width"
│　　│　　　　　　　画像の幅[px]
│　　│　　　　　　　　dtype: np.uint32
│　　│  
│　　├── "[キー2]"
│　　：　　：
│　　：
│
├── "data"
│　　│　データセット本体
│　　│
│　　├── "0"
│　　│　　│　0からの番号
│　　│　　│  
│　　│　　├── "rgb"
│　　│　　│　　　　RGB画像
│　　│　　│　　　　　dtype: np.ndarray<np.uint8>
│　　│　　│　　　　　shape: (h, w, 3)
│　　│　　│　　　　　attribute:
│　　│　　│　　　　　　　type       : "bgr8" or "rgb8"
│　　│　　│　　　　　　　stamp.sec  : タイムスタンプ(整数部)
│　　│　　│　　　　　　　stamp.nsec : タイムスタンプ(小数部)
│　　│　　│　　　　　　　frame_id   : 座標系
│　　│　　│
│　　│　　├── "semantic2d"
│　　│　　│　　　　Semantic Segmentation用ラベル画像
│　　│　　│　　　　　dtype: np.ndarray<np.uint8>
│　　│　　│　　　　　shape: (h, w)
│　　│　　│　　　　　attribute:
│　　│　　│　　　　　　　type       : "semantic2d"
│　　│　　│　　　　　　　stamp.sec  : タイムスタンプ(整数部)
│　　│　　│　　　　　　　stamp.nsec : タイムスタンプ(小数部)
│　　│　　│　　　　　　　frame_id   : 座標系
│　　│　　│
│　　│　　├── "depth"
│　　│　　│　　　　深度マップ[m]
│　　│　　│　　　　　dtype: np.ndarray<np.float32>
│　　│　　│　　　　　shape: (h, w)
│　　│　　│　　　　　attribute:
│　　│　　│　　　　　　　type       : "depth"
│　　│　　│　　　　　　　stamp.sec  : タイムスタンプ(整数部)
│　　│　　│　　　　　　　stamp.nsec : タイムスタンプ(小数部)
│　　│　　│　　　　　　　frame_id   : 座標系
│　　│　　│  
│　　│　　└── "pose"
│　　│　　　　　│　カメラ相対自己位置
│　　│　　　　　│　　attribute:
│　　│　　　　　│　　　　type       : "pose"
│　　│　　　　　│
│　　│　　　　　├── "[キー1]"
│　　│　　　　　│　　│　データセット中のキーに対応
│　　│　　　　　│　　│　　attribute:
│　　│　　　　　│　　│　　　　stamp.sec      : タイムスタンプ(整数部)
│　　│　　　　　│　　│　　　　stamp.nsec     : タイムスタンプ(小数部)
│　　│　　　　　│　　│　　　　frame_id       : 座標系
│　　│　　　　　│　　│　　　　child_frame_id : 座標系(子)
│　　│　　　　　│　　│
│　　│　　　　　│　　├── "translation"
│　　│　　　　　│　　│　　　　並進ベクトル x, y, z
│　　│　　　　　│　　│　　　　　dtype: np.ndarray<np.float64>
│　　│　　　　　│　　│　　　　　shape: (3,)
│　　│　　　　　│　　│　　　　　attribute:
│　　│　　　　　│　　│　　　　　　　type  : "translation"
│　　│　　　　　│　　│　　　　　　　array : "x,y,z"
│　　│　　　　　│　　│
│　　│　　　　　│　　└── "rotation"
│　　│　　　　　│　　　　　　　クォータニオン x, y, z, w
│　　│　　　　　│　　　　　　　　dtype: np.ndarray<np.float64>
│　　│　　　　　│　　　　　　　　shape: (4,)
│　　│　　　　　│　　　　　　　　attribute:
│　　│　　　　　│　　　　　　　　　　type  : "quaternion"
│　　│　　　　　│　　　　　　　　　　array : "x,y,z,w"
│　　│　　　　　│
│　　│　　　　　├── "[キー2]"
│　　│　　　　　：　　：
│　　│　　　　　：
│　　│
│　　├── "1"
│　　：　　：
│　　：
│
├── "label"
│　　│　ラベルの仕様
│　　│
│　　├── "[type]"
│　　│　　│　データtype別にグループを生成
│　　│　　│
│　　│　　├── "0"
│　　│　　│　　│　ラベルの番号
│　　│　　│　　│
│　　│　　│　　├── "name"
│　　│　　│　　│　　　　ラベルの名称
│　　│　　│　　│　　　　　dtype: str
│　　│　　│　　│
│　　│　　│　　└── "color"
│　　│　　│　　　　　　　ラベルの色情報
│　　│　　│　　　　　　　　dtype: np.ndarray<np.uint8>
│　　│　　│　　　　　　　　shape: (3,)
│　　│　　│　　　　　　　　attribute:
│　　│　　│　　　　　　　　　　type  : "color"
│　　│　　│　　　　　　　　　　array : "b,g,r"
│　　│　　│
│　　│　　├── "1"
│　　：　　：　　
│　　：
│
└── "map"
　　　│　地図情報
　　　│
　　　├── "points"
　　　│　　　　三次元点群地図
　　　│　　　　　dtype: np.ndarray<np.float32>
　　　│　　　　　shape: (n, 3)
　　　│　　　　　attribute:
　　　│　　　　　　　type     : "points"
　　　│　　　　　　　frame_id : 座標系
　　　：
　　　：
```
