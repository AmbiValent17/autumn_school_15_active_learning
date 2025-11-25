import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch.optim as optim
import torch
from sklearn.metrics import f1_score, accuracy_score
import torch.nn.functional as F


class ActiveLearning:
  @staticmethod
  def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


  def __init__(self, model, X_train, y_train, X_test, y_test,
               strategy="random", al_type="cumulative",
               update_size=128, init_size=128, skip_size=8, skip=False,
               epochs=1, batch_size=128, logs=False, normalization=True,
               metric="f1", criterion=nn.CrossEntropyLoss()):

    self.model = model
    self.update_size = update_size
    self.batch_size = batch_size
    self.al_type = al_type
    self.criterion = criterion
    self.strategy = strategy
    self.epochs = epochs
    self.metric = metric
    self.skip = skip
    self.logs = logs
    self.normalization = normalization
    self.skip_size = skip_size
    self.most_inf_images = []

    if self.normalization:
       X_train_scaled, X_test_scaled = ActiveLearning.normalize_data(X_train, X_test)

    self.X_init = X_train_scaled[:init_size].clone()
    self.y_init = y_train[:init_size].clone()
    self.X_train = X_train_scaled[:init_size].clone()
    self.y_train = y_train[:init_size].clone()
    self.X_pool = X_train_scaled[init_size:].clone()
    self.y_pool = y_train[init_size:].clone()
    self.X_test = X_test_scaled.clone()
    self.y_test = y_test.clone()

    self.test_metrics = []
    self.labeled_fractions = []
    self.full_size = len(X_train)
    self.labeled_size = len(self.X_init)



  @classmethod
  def normalize_data(cls, X_train, X_test):
    cls.set_seed()
    mean = X_train.float().mean()
    std = X_train.float().std()

    X_train_norm = (X_train.float() - mean) / std
    X_test_norm = (X_test.float() - mean) / std

    return X_train_norm, X_test_norm



  def acquisition_function(self, outputs, strategy):
    self.set_seed()
    match strategy:
      case "random":
        if (self.skip and len(self.X_pool) - self.skip_size > 0):
          indices = torch.randperm(len(self.X_pool))[self.skip_size:self.update_size]
        else:
          indices = torch.randperm(len(self.X_pool))[:self.update_size]

      case"entropy":
        probas = F.softmax(outputs, dim=1)
        entropy = -torch.sum(probas * torch.log(probas + 1e-8), dim=1)
        if (self.skip and len(self.X_pool) - self.skip_size > 0):
          indices = torch.argsort(entropy, descending=True)[self.skip_size:self.update_size]
        else:
          indices = torch.argsort(entropy, descending=True)[:self.update_size]

      case "confidence":
        probas = F.softmax(outputs, dim=1)
        confidence, _ = torch.max(probas, dim=1)
        if (self.skip and len(self.X_pool) - self.skip_size > 0):
          indices = torch.argsort(confidence)[self.skip_size:self.update_size]
        else:
          indices = torch.argsort(confidence)[:self.update_size]

      case "margin":
        probas = F.softmax(outputs, dim=1)
        top2 = torch.topk(probas, 2, dim=1)
        margin = -(top2.values[:, 0] - top2.values[:, 1])
        if (self.skip and len(self.X_pool) - self.skip_size > 0):
          indices = torch.argsort(margin, descending=True)[self.skip_size:self.update_size]
        else:
          indices = torch.argsort(margin, descending=True)[:self.update_size]

      case "dynamic":
        # metrics = []
        # indices = torch.randperm(len(self.X_train))[:self.update_size]
        # X_train_random = self.X_train[indices].clone()
        # y_train_random = self.y_train[indices].clone()
        # y_pred_random = self.predict(X_train_random)
        # if self.metric == "accuracy":
        #   metric_random = accuracy_score(y_train_random.numpy(), y_pred_random.numpy())
        # elif self.metric == "f1":
        #   metric_random = f1_score(y_train_random.numpy(), y_pred_random.numpy(), average="macro")

        # metrics.append(metric_random)

        # outputs_conf = self.predict_proba(self.X_train)
        # confidence, _ = torch.max(probas, dim=1)
        # indices = torch.argsort(confidence)[:self.update_size]
        # X_train_conf = self.X_train[indices].clone()
        # y_train_conf = self.y_train[indices].clone()
        # y_pred_conf = self.predict(X_train_conf)
        # if self.metric == "accuracy":
        #   metric_conf = accuracy_score(y_train_conf.numpy(), y_pred_conf.numpy())
        # elif self.metric == "f1":
        #   metric_conf = f1_score(y_train_conf.numpy(), y_pred_conf.numpy(), average="macro")

        # metrics.append(metric_conf)

        # outputs_margin = self.predict_proba(self.X_train)
        # top2 = torch.topk(outputs_margin, 2, dim=1)
        # margin = -(top2.values[:, 0] - top2.values[:, 1])
        # indices = torch.argsort(margin)[:self.update_size]
        # X_train_margin = self.X_train[indices].clone()
        # y_train_margin = self.y_train[indices].clone()
        # y_pred_margin = self.predict(X_train_margin)
        # if self.metric == "accuracy":
        #   metric_margin = accuracy_score(y_train_margin.numpy(), y_pred_margin.numpy())
        # elif self.metric == "f1":
        #   metric_margin = f1_score(y_train_margin.numpy(), y_pred_margin.numpy(), average="macro")

        # metrics.append(metric_margin)


        # outputs_entropy = self.predict_proba(self.X_train)
        # entropy = -torch.sum(probas * torch.log(probas + 1e-8), dim=1)
        # indices = torch.argsort(entropy)[:self.update_size]
        # X_train_entropy = self.X_train[indices].clone()
        # y_train_entropy = self.y_train[indices].clone()
        # y_pred_entropy = self.predict(X_train_margin)
        # if self.metric == "accuracy":
        #   metric_entropy = accuracy_score(y_train_entropy.numpy(), y_pred_entropy.numpy())
        # elif self.metric == "f1":
        #   metric_entropy = f1_score(y_train_entropy.numpy(), y_pred_entropy.numpy(), average="macro")

        # metrics.append(metric_entropy)
        pass




    X_update = self.X_pool[indices].clone()
    y_update = self.y_pool[indices].clone()

    mask = torch.ones(len(self.X_pool), dtype=torch.bool)
    mask[indices] = False
    self.X_pool = self.X_pool[mask]
    self.y_pool = self.y_pool[mask]
    self.most_inf_images.append((X_update[:16], y_update[:16]))
    return X_update, y_update



  def fit(self, lr=0.01, stop_ratio=1):
    self.set_seed()
    print(f"AL TRAINING STARTED ({self.al_type} {self.strategy})")
    optimizer = optim.Adam(self.model.parameters(), lr=lr)

    pool_size = len(self.X_pool)
    train_size = len(self.X_train)

    while pool_size > 0 and self.update_size > 0:
      n_batches = (train_size + self.batch_size - 1) // self.batch_size
      for epoch in range(self.epochs):
        self.model.train()
        indices = torch.randperm(train_size)

        for batch_idx in range(n_batches):
          start_idx = batch_idx * self.batch_size
          end_idx = min((batch_idx + 1) * self.batch_size, train_size)
          batch_indices = indices[start_idx:end_idx]

          X_batch = self.X_train[batch_indices]
          y_batch = self.y_train[batch_indices]

          optimizer.zero_grad()
          outputs = self.model(X_batch)
          loss = self.criterion(outputs, y_batch)
          loss.backward()
          optimizer.step()

      with torch.no_grad():
        self.model.eval()
        pool_outputs = self.model(self.X_pool)
        test_outputs = self.model(self.X_test)
        test_preds = torch.argmax(test_outputs, dim=1)

        match self.metric:
          case "accuracy":
            test_score = accuracy_score(self.y_test.numpy(), test_preds.numpy())
          case "f1":
            test_score = f1_score(self.y_test.numpy(), test_preds.numpy(), average="macro")

        self.test_metrics.append(test_score)
        self.labeled_fractions.append((self.labeled_size / self.full_size) * 100)

        if (self.labeled_fractions[-1] > stop_ratio * 100):
           print(f"AL TRAINING FINISHED ({self.al_type} {self.strategy})")
           return

        X_update, y_update = self.acquisition_function(pool_outputs, self.strategy)

        if self.al_type == "incremental":
          self.X_train = X_update.clone()
          self.y_train = y_update.clone()
          train_size = len(self.X_train)

        if self.al_type == "half-cumulative":
          self.X_train = torch.cat([self.X_train, X_update])
          self.y_train = torch.cat([self.y_train, y_update])
          train_size = len(self.X_train)

        if self.al_type == "cumulative":
          self.X_train = torch.cat([self.X_train, X_update])
          self.y_train = torch.cat([self.y_train, y_update])
          train_size = len(self.X_train)
          for layer in self.model.children():
              if hasattr(layer, 'reset_parameters'):
                  layer.reset_parameters()

        self.labeled_size += len(X_update)

        pool_size = len(self.X_pool)
        if self.logs:
            print(f"Training AL: {self.al_type} {self.strategy} | Доля: {self.labeled_fractions[-1]:.3f}")
    print(f"AL TRAINING FINISHED ({self.al_type} {self.strategy})")
    print()


  def predict_proba(self, X):
    self.model.eval()
    with torch.no_grad():
        return F.softmax(self.model(X), dim=1)

  def predict(self, X):
      self.model.eval()
      with torch.no_grad():
          logits = self.model(X)
          return torch.argmax(logits, dim=1)
      

class ANN(nn.Module):
  def __init__(self, input_dim=28*28, hidden_dim=128, output_dim=10):
    super().__init__()
    self.flat = nn.Flatten()
    self.lin1 = nn.Linear(input_dim, hidden_dim)
    self.relu = nn.ReLU()
    self.lin2 = nn.Linear(hidden_dim, output_dim)
    self.hidden_dim = hidden_dim

  def forward(self, X):
    X = self.flat(X)
    X = self.lin1(X)
    X = self.relu(X)
    logits = self.lin2(X)
    return logits

  def predict_proba(self, X):
    logits = self(X)
    return F.softmax(logits)
  

def plot_active_learning_results_many(*al_objects, dataset_name="", title=None):
    plt.figure(figsize=(12, 8))

    for al_obj in al_objects:
      if al_obj.al_type == "incremental":
        al_type_name = "inc"
      elif al_obj.al_type == "cumulative":
        al_type_name = "cumul"
      elif al_obj.al_type == "half-cumulative":
        al_type_name = "half-cumul"

      hidden_dim = al_obj.model.hidden_dim

      legend_label = f"{al_type_name} {al_obj.strategy} (h{hidden_dim}_e{al_obj.epochs}{f"_s{al_obj.skip_size}" if al_obj.skip else ""})"

      plt.plot(al_obj.labeled_fractions, al_obj.test_metrics,
               marker='o', linewidth=0.5, markersize=1, label=legend_label)

    plt.xlabel('Доля размеченных данных (%)', fontsize=12)
    if al_objects[0].metric == "accuracy":
      plt.ylabel('Accuracy', fontsize=12)
    else:
      plt.ylabel('F1 Score', fontsize=12)

    if title:
        plt.title(title, fontsize=14, pad=20)
    else:
        plt.title(f'Active Learning Results ({dataset_name})', fontsize=14, pad=20)

    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    first_obj = al_objects[0]
    params_text = f"batch_size={first_obj.batch_size}, init_size={first_obj.X_init.shape[0]}, update_size={first_obj.update_size}"

    plt.figtext(0.5, 0.01, params_text, ha='center', fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    plt.show()




import base64
import io
import json
import uuid
import numpy as np
import torch
from PIL import Image
from IPython.display import display, HTML

# --- Безопасная кодировка данных ---
class SafeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if torch.is_tensor(obj):
            if obj.numel() == 1: return obj.item()
            return obj.detach().cpu().numpy().tolist()
        return super(SafeEncoder, self).default(obj)

def tensor_to_base64_safe(tensor, size=(56, 56)):
    """Преобразует тензор в Base64 с адаптивным ресайзом."""
    try:
        if not torch.is_tensor(tensor):
            tensor = torch.tensor(tensor)

        img_np = tensor.detach().cpu().numpy()

        if img_np.ndim == 3 and img_np.shape[0] == 1:
            img_np = img_np.squeeze(0)
        elif img_np.ndim == 3 and img_np.shape[0] > 3:
             img_np = img_np.mean(axis=0)

        img_min, img_max = img_np.min(), img_np.max()
        if img_max > img_min:
            img_np = (img_np - img_min) / (img_max - img_min)

        img_np = (img_np * 255).astype(np.uint8)

        image = Image.fromarray(img_np)
        image = image.resize(size, Image.NEAREST)

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    except Exception:
        return ""

def plot_active_learning_results(al, dataset_name="Dataset", title=None):
    unique_id = uuid.uuid4().hex
    chart_id = f"chart_{unique_id}"

    data_points = []

    if hasattr(al, 'labeled_fractions') and hasattr(al, 'test_metrics'):
        history_len = len(al.labeled_fractions)

        # Адаптивное сжатие для производительности
        if history_len > 150:
            img_size = (28, 28)
        elif history_len > 50:
            img_size = (42, 42)
        else:
            img_size = (56, 56)

        for i in range(history_len):
            fraction = float(al.labeled_fractions[i])

            raw_score = al.test_metrics[i]
            if torch.is_tensor(raw_score):
                score = raw_score.item()
            elif isinstance(raw_score, (np.floating, np.integer)):
                score = raw_score.item()
            else:
                score = raw_score

            sample_count = int(round((fraction * al.full_size) / 100))

            point_data = {
                "fraction": round(fraction, 2),
                "score": round(score, 4),
                "count": sample_count,
                "images": []
            }

            img_batch_idx = i - 1

            if 0 <= img_batch_idx < len(al.most_inf_images):
                data_tuple = al.most_inf_images[img_batch_idx]

                if isinstance(data_tuple, (list, tuple)):
                    images_tensor = data_tuple[0]
                else:
                    images_tensor = data_tuple

                limit = min(10, len(images_tensor))
                for k in range(limit):
                    b64 = tensor_to_base64_safe(images_tensor[k], size=img_size)
                    if b64:
                        point_data["images"].append(f"data:image/png;base64,{b64}")

            data_points.append(point_data)

    json_data = json.dumps(data_points, cls=SafeEncoder)

    display_title = title if title else f"Active Learning: {dataset_name}"
    metric_label = "F1 Score" if hasattr(al, 'metric') and al.metric == "f1" else "Accuracy"

    # Генерируем полный HTML документ
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>{display_title}</title>
    <script src="https://cdn.jsdelivr.net/npm/echarts@5.4.3/dist/echarts.min.js"></script>
    <style>
        body {{ margin: 0; padding: 20px; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif; background: #f0f2f5; }}
        .chart-wrapper {{
            width: 100%;
            max_width: 1400px;
            margin: 0 auto;
            background: #fff;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            overflow: hidden;
            position: relative;
        }}
        /* ФИКСИРОВАННАЯ ВЫСОТА, КАК ПРОСИЛИ */
        #{chart_id} {{
            width: 100%;
            height: 750px;
        }}
        #tooltips-layer {{
            position: absolute; top: 0; left: 0; width: 100%; height: 100%; z-index: 10; pointer-events: none;
        }}
        /* Стили всплывающего окна */
        .custom-tooltip {{
            position: absolute;
            width: 280px;
            background: rgba(255, 255, 255, 0.98);
            border: 1px solid #dce1e6;
            border-radius: 8px;
            box-shadow: 0 8px 24px rgba(0,0,0,0.12);
            display: flex;
            flex-direction: column;
            pointer-events: auto;
            z-index: 100;
            font-size: 13px;
            color: #333;
            transition: opacity 0.1s;
        }}
        .tip-header {{
            padding: 10px 14px;
            background: #f4f6f8;
            border-bottom: 1px solid #e1e4e8;
            border-radius: 8px 8px 0 0;
            display: flex;
            justify-content: space-between;
            align-items: center;
            cursor: grab;
            font-weight: 600;
            color: #3390ec;
        }}
        .tip-header:active {{ cursor: grabbing; }}
        .close-btn {{ cursor: pointer; font-size: 18px; color: #888; line-height: 1; }}
        .close-btn:hover {{ color: #d32f2f; }}
        .tip-body {{ padding: 14px; }}
        .row {{ display: flex; justify-content: space-between; margin-bottom: 6px; }}
        .img-grid {{ display: grid; grid-template-columns: repeat(5, 1fr); gap: 5px; margin-top: 12px; }}
        .img-grid img {{ width: 100%; border-radius: 4px; border: 1px solid #eee; pointer-events: none; }}
    </style>
</head>
<body>

    <div class="chart-wrapper">
        <div id="{chart_id}"></div>
        <div id="tooltips-layer"></div>
    </div>

    <script>
    (function() {{
        var chartDom = document.getElementById('{chart_id}');
        var myChart = echarts.init(chartDom);
        var rawData = {json_data};

        var xData = rawData.map(d => d.fraction);
        var yValues = rawData.map(d => d.score);

        // Функция стилей (Обычные точки - маленькие и сплошные, Выделенные - большие и оранжевые)
        function getSeriesData(pinnedMap) {{
            return yValues.map((val, idx) => {{
                var isPinned = pinnedMap[idx];
                if (isPinned) {{
                    return {{
                        value: val,
                        symbol: 'circle',
                        symbolSize: 18,
                        itemStyle: {{
                            color: '#ff8800',
                            borderColor: '#fff',
                            borderWidth: 3,
                            shadowBlur: 10,
                            shadowColor: 'rgba(255, 136, 0, 0.5)'
                        }}
                    }};
                }} else {{
                    return {{
                        value: val,
                        symbol: 'circle',
                        symbolSize: 6,  // Маленькие точки
                        itemStyle: {{
                            color: '#3390ec', // Сплошной цвет без белой середины
                            borderColor: '#3390ec',
                            borderWidth: 0
                        }}
                    }};
                }}
            }});
        }}

        var option = {{
            title: {{ text: '{display_title}', left: 'center', top: 15, textStyle: {{ fontSize: 20, color: '#222' }} }},
            grid: {{ left: 60, right: 40, bottom: 90, top: 70, containLabel: true }},
            tooltip: {{ show: false }},

            dataZoom: [
                {{ type: 'slider', show: true, xAxisIndex: [0], bottom: 30, height: 25 }},
                {{ type: 'inside', xAxisIndex: [0] }}
            ],

            xAxis: {{
                type: 'category',
                name: 'Labeled Data (%)',
                nameLocation: 'middle',
                nameGap: 40,
                data: xData,
                boundaryGap: false,
                axisLine: {{ lineStyle: {{ color: '#ccc' }} }},
                axisLabel: {{ color: '#666' }}
            }},
            yAxis: {{
                type: 'value',
                name: '{metric_label}',
                nameLocation: 'middle',
                nameGap: 50,
                scale: true,
                splitLine: {{ lineStyle: {{ type: 'dashed', color: '#eee' }} }}
            }},
            series: [{{
                type: 'line',
                data: getSeriesData({{}}),
                smooth: true,
                showAllSymbol: true,
                lineStyle: {{ width: 3, color: '#3390ec' }},
                areaStyle: {{
                    color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
                        {{ offset: 0, color: 'rgba(51, 144, 236, 0.25)' }},
                        {{ offset: 1, color: 'rgba(51, 144, 236, 0.0)' }}
                    ])
                }},
                animationDurationUpdate: 100
            }}]
        }};

        myChart.setOption(option);

        // --- Логика Тултипов ---
        var tooltipsLayer = document.getElementById('tooltips-layer');
        var pinnedState = {{}};

        function updateChartStyle() {{
            myChart.setOption({{ series: [{{ data: getSeriesData(pinnedState) }}] }});
        }}

        myChart.on('click', function(params) {{
            if (params.componentType !== 'series') return;

            var idx = params.dataIndex;
            var tipId = 'tooltip_' + idx;

            // Закрытие если открыто
            if (pinnedState[idx]) {{
                var existing = document.getElementById(tipId);
                if (existing) existing.remove();
                delete pinnedState[idx];
                updateChartStyle();
                return;
            }}

            var item = rawData[idx];
            var point = myChart.convertToPixel('grid', [xData[idx], yValues[idx]]);
            var px = point[0];
            var py = point[1];
            var chartW = myChart.getWidth();

            // Изображения
            var imgHtml = '';
            if (item.images && item.images.length > 0) {{
                imgHtml = '<div class="img-grid">';
                item.images.forEach(src => {{
                    imgHtml += '<img src="' + src + '">';
                }});
                imgHtml += '</div>';
            }} else {{
                imgHtml = '<div style="text-align:center;color:#999;margin-top:10px;">No images</div>';
            }}

            var tip = document.createElement('div');
            tip.id = tipId;
            tip.className = 'custom-tooltip';
            tip.innerHTML = `
                <div class="tip-header">
                    <span>Step ${{idx}}</span>
                    <span class="close-btn">&times;</span>
                </div>
                <div class="tip-body">
                    <div class="row"><span>{metric_label}:</span> <b>${{item.score}}</b></div>
                    <div class="row"><span>Data:</span> <b>${{item.fraction}}% (${{item.count}})</b></div>
                    ${{imgHtml}}
                </div>
            `;

            tooltipsLayer.appendChild(tip);

            // Позиционирование (Центрирование)
            var tipW = 282;
            var tipH = tip.offsetHeight;
            var left = px - (tipW / 2);

            // Границы
            if (left < 10) left = 10;
            if (left + tipW > chartW) left = chartW - tipW - 10;

            tip.style.left = left + 'px';

            // Верх/Низ
            if (py > 400) {{ // Если точка ниже середины (условно)
                tip.style.top = (py - tipH - 20) + 'px';
            }} else {{
                tip.style.top = (py + 20) + 'px';
            }}

            // Закрытие
            tip.querySelector('.close-btn').onclick = function() {{
                tip.remove();
                delete pinnedState[idx];
                updateChartStyle();
            }};

            // Drag & Drop
            var header = tip.querySelector('.tip-header');
            header.onmousedown = function(e) {{
                e.preventDefault();
                var shiftX = e.clientX - tip.getBoundingClientRect().left;
                var shiftY = e.clientY - tip.getBoundingClientRect().top;
                var containerRect = tooltipsLayer.getBoundingClientRect();

                function moveAt(pageX, pageY) {{
                    tip.style.left = (pageX - shiftX - containerRect.left) + 'px';
                    tip.style.top = (pageY - shiftY - containerRect.top) + 'px';
                }}

                function onMouseMove(e) {{ moveAt(e.pageX, e.pageY); }}
                document.addEventListener('mousemove', onMouseMove);
                document.onmouseup = function() {{
                    document.removeEventListener('mousemove', onMouseMove);
                    document.onmouseup = null;
                }};
            }};
            header.ondragstart = function() {{ return false; }};

            pinnedState[idx] = true;
            updateChartStyle();
        }});

        window.addEventListener('resize', function() {{ myChart.resize(); }});
    }})();
    </script>
</body>
</html>
"""

    # 1. Отображение в ноутбуке
    display(HTML(html_content))

    # 2. Сохранение и скачивание файла
    filename = "active_learning_chart.html"
    with open(filename, "w") as f:
        f.write(html_content)

    print(f"✅ График сохранен как '{filename}'. Скачивание начнется автоматически...")
