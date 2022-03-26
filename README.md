# University-field-training

### (주) 로보스타 현장실습 진행
 * 하계 현장실습으로 해당 기업에서 매뉴얼 작성 및 데이터 시각화 업무를 수행했다.


### 깃허브 파일 설명
(그래프를 확인하기 위해 [Open in Colab] 이용하는 것을 권장합니다.)

 * robostar_column_change.ipynb : 각 데이터를 특정 프로세스에 따라 처리한 후 bokeh 라이브러리를 이용하여 시각화

 * python_bokeh.ipynb : bokeh 라이브러리를 활용한 데이터 시각화 방법을 공유하기 위해 작성

 * tensorflow.ipynb : Model_Fitting_Deeplearning.ipynb를 작성하기 위해 데이터 분석 및 텐서플로우 실행 예제 

 * Model_Fitting_Deeplearning.ipynb : 실제 데이터와 curve_fit으로 예측한 데이터, keras를 이용하여 예측한 데이터를 비교하기 위해 bokeh 라이브러리를 이용하여 시각화, 각 그래프를 ppt 특정 위치에 지정 후 파일 저장


### 진행 과정
 * bokeh 라이브러리를 처음 사용하는 만큼 [Documentation](https://docs.bokeh.org/en/latest/#) 및 [Tutorials](https://nbviewer.org/github/bokeh/bokeh-notebooks/blob/master/tutorial/00%20-%20Introduction%20and%20Setup.ipynb)를 참고했다.
 
 * bokeh의 특징인 디자인과 반응형을 최대한 반영할 수 있도록 노력했다.

 * 데이터 분석에 도움이 될 수 있는 다양한 기능을 추가했다.
 
 * Colab 환경에서 개발을 해야 했으므로 Jupyter에서만 가능한 기능은 구현하지 못했다.


### 결과
1. robostar_column_change.ipynb - 각 데이터를 프로세스에 따라 처리한 후 bokeh 라이브러리를 이용하여 시각화

2. python_bokeh.ipynb - robostar_column_change.ipynb에서 그래프를 그릴때마다 bokeh의 레전드를 설정해야 하는 번거로움을 줄이기 위해 make_graphs 함수를 생성했다. make_graphs를 통해 수치 데이터를 그래프로 쉽게 변환할 수 있도록 작성했다.

``` python
def make_graphs(log_datas, y_axis, x_axis, title, x_axis_label, y_axis_label, size, alpha):
```

log_datas : 동일한 양식의 df가 저장되어 있는 리스트 형태  
y_axis, x_axis : y축 데이터 이름 리스트, x축 데이터 이름 설정  
(y_axis의 경우 여러 데이터가 입력되기 때문에 list형태로 구성)  
title : 그래프의 타이틀 설정  
x_axis_label, y_axis_label = x축, y축 이름 설정  
size : 그래프 원의 크기 조정  
alpha : 그래프 원의 투명도 조정  

``` python
for j in range(0, len(y_axis)):
      # 색깔 지정을 위한 if, else
      if j == 0:
        p.extra_y_ranges = {y_axis[j] : Range1d(start=min(log_datas[i][y_axis[j]]), end=max(log_datas[i][y_axis[j]]))}
        p.circle(log_datas[i][x_axis], log_datas[i][y_axis[j]], y_range_name=y_axis[j], line_color=palette[j], size=size, alpha=alpha)

      else:
        p.extra_y_ranges[y_axis[j]] = Range1d(start=min(log_datas[i][y_axis[j]]), end=max(log_datas[i][y_axis[j]]))
        if j % 2 == 0:
          p.circle(log_datas[i][x_axis], log_datas[i][y_axis[j]], y_range_name=y_axis[j], line_color=palette[j], size=size, alpha=alpha)
          p.add_layout(LinearAxis(y_range_name=y_axis[j], axis_label=y_axis[j], axis_label_text_color=palette[j], axis_line_color=palette[j]), "right")
        else :
          p.circle(log_datas[i][x_axis], log_datas[i][y_axis[j]], y_range_name=y_axis[j], line_color=palette[-j], size=size, alpha=alpha)
          p.add_layout(LinearAxis(y_range_name=y_axis[j], axis_label=y_axis[j], axis_label_text_color=palette[-j], axis_line_color=palette[j]), "right")
```

p.extra_y_ranges를 통해 y축의 size를 조정한다.  
(미수행 시 큰 값과 작은 값이 같이 그려 질 경우 작은 값 분석이 어려울 수 있음)  
그래프 첫 수행 시 p.circle()을 통해 바로 그릴 수 있지만 이후에는 p.add_layout()을 통해 그래프에 추가시켜야 한다.  
(j를 2로 나눈 나머지 값으로 조건을 준 것은 팔레트를 사용할 경우 연속된 값의 색깔은 유사하기 때문에 연속된 색깔을 사용하지 않도록 구성한 것)  


``` python
    p.legend.orientation = "horizontal"
    p.legend.location = "top_right"
    p.legend.click_policy = "hide"
    p.background_fill_color="#f0f0f0"
    p.grid.grid_line_color="#f5f5f5"
    p.toolbar.active_scroll = p.select_one(WheelZoomTool)
```
legend의 위치, legend 클릭 시 수행할 기능, 배경 색, 배경 그리드 색, 스크롤 시 수행할 기능을 추가한다

3. tensorflow.ipynb - 기존 데이터 예측 모델인 curve_fit 대신 keras를 이용하여 새로운 예측 모델을 구성하기 위해 데이터를 분석

4. Model_Fitting_Deeplearning.ipynb - tensorflow.ipynb에서 데이터를 분석하고 keras를 이용해 새로운 예측 모델을 구성하고 분석한 결과 유의미한 결과가 있다고 생각되어 실제 데이터와 curve_fit, keras를 비교한 그래프를 생성 및 분석

``` python

    model = build_model(train_dataset)
    EPOCHS = 5000
    early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=100)
    history = model.fit(
    normed_train_data, train_labels,
    epochs=EPOCHS, validation_split=0.2, verbose=0, callbacks=[early_stop])

```
keras의 가장 기초적인 모델을 구성했다.

