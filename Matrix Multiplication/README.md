## Параметры системы
Программа запускалась в среде разработки Google Colab (в виду отсутствия видеокарты с Nvidia)

CPU: Intel(R) Xeon(R) CPU @ 2.30GHz 13 ГБ RAM

GPU: Tesla K80  CUDA Version: 11.2 

## Входные данные

BLOCK_SIZE = 16   - колицество нитей в блоке

N = 1024 - размер матриц (размерность матриц кратна количетсву нитей в блоке для повышения эффективности алгоритма)

## Тестовый запуск 

![image](https://user-images.githubusercontent.com/50167514/143007913-9e883cef-3d2d-4f82-97ec-3f0512ccd9b0.png)

## Сравнение результатов на разных размерностях 

| n  | 128 | 256 | 512 | 1024 | 2046 |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| gpuTime, ms | 0,16  |	0,39  |	1,53  |	4,24  |	16,38  |
| cpuTime, ms  | 8068  |	82783  |	747805  |	6129267  |	63504839   |
| acceleration   | 50425  |	212264,1026  |	488761,4379  |	1445581,84  |	3876974,298  |


![image](https://user-images.githubusercontent.com/50167514/143008252-398a4760-15b6-4262-b4ac-d759e79046c5.png)

*немного смущают большие числа cpuTime, но по идее все считается правильно

## Вывод

Программа на CUDA (GPU) работает быстрее, чем программа на CPU. Особенно на хорошей видеокарте. График роста ускорения наглядно это демонстрирует 
