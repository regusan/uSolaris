# TinyReguRender

## ビルド

```bash
cmake -B build -S . -DCMAKE_CXX_COMPILER=g++-12
cmake --build build
```

## examples/bmp の実行

```bash
./build/examples/bmp/example_bmp
```


mesh
{
    verticies[N_v+edge]#頂点の重複を覚悟して、[material1で参照する頂点,2で参照する頂点...]のようにmaterial1の処理時は"material1で参照する頂点"のみをロードすればよくする
    indicies[N_i[3]]
    materialrange = {[0,N_m1],[N_m1,N_m2],[N_m1+N_m2],...}  
}

instancedmesh
{
    mesh *cheir;
    materialslot = [metal,wood,wood2]
}

for material
    for mesh
        for indicies　
            transform

mesh
{
    verticies[N_v+edge]#頂点の重複を覚悟して、[material1で参照する頂点,2で参照する頂点...]のようにmaterial1の処理時は"material1で参照する頂点"のみをロードすればよくする
    indicies[N_i[3]]
    meshlets{
        {vertstart,vertend,},{vertstart,vertend},
    }
    materialrange = {[0,N_m1],[N_m1,N_m2],[N_m1+N_m2],...}  
}

instancedmesh
{
    mesh *cheir;
    materialslot = [metal,wood,wood2]
}

for material
    for mesh
        for indicies
            transform