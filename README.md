example classificaiton server with fastapi and tensorflow serving.

## how to build

- put sentencepiece model files into fastapi/model
- put exported tensorflow SavedModel into servingtf/export
- put labels.txt file into fastapi/app


## todo

- 開発時と本番時の設定切り替え
  - --reload flag
  - 本番時はvolume mountいらない
- imageの小型化( fastapiの方, alpineのimage使おうとしたらpip install sentencepieceのとこで落ちた )
