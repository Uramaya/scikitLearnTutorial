# ScrapyTutorial
# CYO�������݉�A�v��


## �����ݒ�
* �J���v���W�F�N�g��git clone����
``` bash
�@git clone ssh://git@gitlab.avelio.jp:60122/CYOFindFoodApp/scrapytutorial.git 
```

* Docker�}�V�����N��
�@
```bash
�@docker-machine start
```
�@
�@
* Docker�}�V�����N���m�F
�@
```bash
�@docker-machine ls
```

* Docker�R���e�i���r���h�A�����グ��
``` bash
�@docker-compose up
```

* �V�����^�[�~�i�����J���ADocker�R���e�i������ɋN���������m�F
``` bash
�@docker-compose ps 
```
�i��j
```
           Name              Command   State   Ports
----------------------------------------------------
testenv_your-sevice-name_1   python3   Up


```

* �R���e�i�ɓ���

``` bash
winpty docker exec -it testenv_your-sevice-name_1 bash
```

* �R���e�i�ɓ�������Apython���s�A�uHello World�v���ĕ\�����ꂽ�琬��

``` bash
python test.py
```


