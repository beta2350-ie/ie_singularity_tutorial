# Singularityを用いたFCN_RESNET101のセマンティック・セグメンテーション

学科サーバに搭載されたGPUをSingularityを用いて使用する方法に関するメモ

<!--
## python環境構築
- pythonのバージョンを設定
    - pyenvなどを使うと便利(入れ方に関しては調べるとすぐ出てくる)
    ```
        #入れたいPythonバージョンを探す
        pyenv install --list
        #Pythonをインストールする
        pyenv install 3.x.x
        #Pythonバージョンをローカルに反映
        pyenv local 3.x.x
        #仮想環境を構築
        python -m venv venv
        #仮想環境を実行
        source venv/bin/activate
    ```
よく考えたら、singularity内に環境作るから、ここで作る必要ない。
-->
- singularity用のイメージをダウンロードする
    - どのパッケージで学習させるかによる。(とりあえずtensorflowとpytorchは確認済み)
    - CUDAに対応したDocker Imageをpullしてくる。
        - Docker Imageを使用することでsingularity Imageとして使用できる。
        - tensorflow
            - https://hub.docker.com/r/tensorflow/tensorflow/tags?page=1
            - xxx-gpu-py3みたいなバージョンを選べばいいらしい。(latest-gpu-py3でいけるらしい)
        - pytorch
            - https://hub.docker.com/r/pytorch/pytorch
            - こっちはlatestでいける。  
    - pull(pytorchの場合)
        - `singularity pull docker://pytorch/pytorch:latest`
        - カレントディレクトリに`xxx.sif`が生成される。　

- Definition Fileを作成
    - 必要なパッケージをinstallする(aptとかpipとか)
        - パッケージのバージョン関係は結構厳しいらしいので、地道にチェックする必要がある。
    ```
        BootStrap: localimage                                                                                   
        From: xxx.sif(さっきダウンロードしたsifファイル)

        %post
            apt-get update
            apt-get install -y \
                vim \
                wget \
                make \
                curl
        # 必要なパッケージを追加

            pip3 install --upgrade pip
        # 必要なpythonパッケージを追加
            pip3 install tqdm numpy matplotlib seaborn
    ```

- Definition Fileからsifファイルを作成
    - `singularity build --fakeroot hoge.sif huga.def`
        - hoge.sifは作成したいsifファイル名
        - huga.defはDefinition File名

- singularityを実行する。
    - `singularity exec --nv hoge.sif python fcn_resnet101.py`
    - GPUを独占してしまうので、早めに実行を終了する。

- slurmを用いてジョブ管理をする。
    - sifファイルは絶対パスか相対パスで指定。
    - `yyy.sbatch`を作成する。
    ```
        #!/bin/bash                                                                                                           
        #SBATCH --job-name 自由なジョブ名を指定
        #SBATCH --output logs/%x-%j.log
        #SBATCH --error logs/%x-%j.err
        #SBATCH --nodes 1
        #SBATCH --gpus tesla:1

        date
        singularity exec --nv hoge.sif python fcn_resnet101.py
        date
    ```
    - slurmにジョブを追加
        - `sbatch yyy.sbatch`
    - ジョブの追加を確認
        - `squeue`
    - ジョブの削除
        - `scancel ジョブID`
        - ジョブIDはsqueueで確認可能。

- 出力の確認
    - 出力はlogsファイルに生成される。
    - 正常なlogは`ジョブ名-ジョブID.log`で出力。
    - 異常なlogは`ジョブ名-ジョブID.err`で出力。

- 使うと便利なコマンド
    - watch -n 1 squeue
        - squeueは実行したときの情報しか出さないが、watchコマンドを使うことで1秒間隔で実行できる。
        - sbatchが実行中か終了したかがわかるため、結構便利。
    - watch -n 1 nvidia-smi
        - GPUの情報を出力してくれるコマンド。
        - GPUの仕様のほか、現在の使用メモリや消費電力、実行中のプロセスなどが見れる。
        - 現在の処理がどの程度メモリを使用しているかを確認できる。
            - ~~他人の金でどの程度、電気を使用しているかを確認できる。~~

## 参照
- nal先生のチュートリアル
    - https://github.com/naltoma/trial-keras-bert
- ieのチュートリアル
    - https://ie.u-ryukyu.ac.jp/syskan/service/singularity/
- kaggleのcat and dog データセット
    - https://www.kaggle.com/datasets/tongpython/cat-and-dog
