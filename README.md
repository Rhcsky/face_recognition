# Face recognition using few shot learning

This is the ***PARAN semester*** project in Ajou university in 2021 spring.

The double relation few-shot model can recognition someone's face faster than other deep learning model and the model
size is very small. You can use this model only 3 step.

### 1. Register your face in facebank

We call face database a `facebank` and it has the following folder structure.

```
facebank
`-- images
    |-- hankyul
    |   |-- 1.jpg
    |   `-- 2.jpg
    |-- sanghyun
    |   |-- 1.jpg
    |   `-- 2.jpg
    `-- seungmin
        |-- 1.jpg
        `-- 2.jpg
```

In facebank, you can register someone's class and face.

### 2. Make facebank vector

For fast face recognition, you have to make `facebank vector` before starting the inference. But don't worry about make
that. You just run below command. The `facebank_vector` will be saved in facebank folder.

```python
python
make_facebank.py
```

### 3. Run inference

Now, every preprocessing is ended. you enter the image_path at `infer.py`. That's it!

```python
python
infer.py
```



