

```python
#import packages
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import statistics
import seaborn as sns 

df = pd.read_csv(r"C:\Users\ymf\Desktop\Onomy\coursera-course-detail-data.csv")
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>Name</th>
      <th>Url</th>
      <th>Rating</th>
      <th>Difficulty</th>
      <th>Tags</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>"Making" Progress Teach-Out</td>
      <td>https://coursera.org/learn/makingprogress</td>
      <td>None</td>
      <td>Beginner Level</td>
      <td>['Social Sciences', 'Governance and Society']</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>(Business Writing) الكتابة في مجال الأعمال</td>
      <td>https://coursera.org/learn/writing-for-busines...</td>
      <td>None</td>
      <td>Beginner Level</td>
      <td>['Business', 'Business Essentials']</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>(Giving Helpful Feedback) إعطاء الملاحظات المفيدة</td>
      <td>https://coursera.org/learn/feedback-ar</td>
      <td>4.8</td>
      <td>None</td>
      <td>['Business', 'Business Essentials']</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>(Successful Presentation) العرض التقديمي الناجح</td>
      <td>https://coursera.org/learn/presentation-skills-ar</td>
      <td>4.9</td>
      <td>Beginner Level</td>
      <td>['Business', 'Business Essentials']</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>.Net平台下的软件开发技术毕业项目</td>
      <td>https://coursera.org/learn/net-ruanjian-kaifa-...</td>
      <td>None</td>
      <td>None</td>
      <td>['Computer Science', 'Software Development']</td>
    </tr>
  </tbody>
</table>
</div>




```python
df = df[df.Rating != "None"]
df = df[df.Difficulty != "None"]
df = df[df.Difficulty == "Beginner Level"]
df = df[df.Name.str.contains(r'[professional]')]
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>Name</th>
      <th>Url</th>
      <th>Rating</th>
      <th>Difficulty</th>
      <th>Tags</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>(Successful Presentation) العرض التقديمي الناجح</td>
      <td>https://coursera.org/learn/presentation-skills-ar</td>
      <td>4.9</td>
      <td>Beginner Level</td>
      <td>['Business', 'Business Essentials']</td>
    </tr>
    <tr>
      <th>7</th>
      <td>7</td>
      <td>20世纪西方音乐 Western Music in the 20th Century</td>
      <td>https://coursera.org/learn/20cnwm</td>
      <td>4.3</td>
      <td>Beginner Level</td>
      <td>['Arts and Humanities', 'Music and Art']</td>
    </tr>
    <tr>
      <th>12</th>
      <td>12</td>
      <td>3D CAD Fundamental</td>
      <td>https://coursera.org/learn/3d-cad-fundamental</td>
      <td>4.8</td>
      <td>Beginner Level</td>
      <td>['Physical Science and Engineering', 'Mechanic...</td>
    </tr>
    <tr>
      <th>17</th>
      <td>17</td>
      <td>Comercio, Inmigración y Tipos de Cambio en un...</td>
      <td>https://coursera.org/learn/comercio-inmigracio...</td>
      <td>5</td>
      <td>Beginner Level</td>
      <td>['Social Sciences', 'Economics']</td>
    </tr>
    <tr>
      <th>18</th>
      <td>18</td>
      <td>FPGA computing systems: Background knowledge a...</td>
      <td>https://coursera.org/learn/fpga-intro</td>
      <td>4.6</td>
      <td>Beginner Level</td>
      <td>['Computer Science', 'Design and Product']</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>3826</th>
      <td>3826</td>
      <td>Les deux infinis et nous - Voyages de l'infini...</td>
      <td>https://coursera.org/learn/physique-2-infinis-...</td>
      <td>4.8</td>
      <td>Beginner Level</td>
      <td>['Physical Science and Engineering', 'Physics ...</td>
    </tr>
    <tr>
      <th>3833</th>
      <td>3833</td>
      <td>AstroTech: The Science and Technology behind A...</td>
      <td>https://coursera.org/learn/astronomy-technology</td>
      <td>4.7</td>
      <td>Beginner Level</td>
      <td>['Physical Science and Engineering', 'Physics ...</td>
    </tr>
    <tr>
      <th>3834</th>
      <td>3834</td>
      <td>Cybersecurity and Its Ten Domains</td>
      <td>https://coursera.org/learn/cyber-security-domain</td>
      <td>4.2</td>
      <td>Beginner Level</td>
      <td>['Computer Science', 'Computer Security and Ne...</td>
    </tr>
    <tr>
      <th>3837</th>
      <td>3837</td>
      <td>The History of Modern Israel – Part I: From an...</td>
      <td>https://coursera.org/learn/history-israel</td>
      <td>4.5</td>
      <td>Beginner Level</td>
      <td>['Arts and Humanities', 'History']</td>
    </tr>
    <tr>
      <th>3839</th>
      <td>3839</td>
      <td>Mandarin Chinese 3: Chinese for Beginners</td>
      <td>https://coursera.org/learn/mandarin-chinese-3</td>
      <td>4.9</td>
      <td>Beginner Level</td>
      <td>['Language Learning', 'Other Languages']</td>
    </tr>
  </tbody>
</table>
<p>1257 rows × 6 columns</p>
</div>




```python
personal = df[df['Tags'].str.contains("Personal")|df['Name'].str.contains("Personal")]
print(personal.shape)
personal.head()
```

    (58, 6)
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>Name</th>
      <th>Url</th>
      <th>Rating</th>
      <th>Difficulty</th>
      <th>Tags</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>56</th>
      <td>56</td>
      <td>Competencias digitales. Bases de datos: Access</td>
      <td>https://coursera.org/learn/competencias-digita...</td>
      <td>4.7</td>
      <td>Beginner Level</td>
      <td>['Personal Development', 'Personal Development']</td>
    </tr>
    <tr>
      <th>64</th>
      <td>64</td>
      <td>The Science of Training Young Athletes</td>
      <td>https://coursera.org/learn/youth-sports</td>
      <td>4.8</td>
      <td>Beginner Level</td>
      <td>['Personal Development', 'Personal Development']</td>
    </tr>
    <tr>
      <th>108</th>
      <td>108</td>
      <td>Tinkering Fundamentals: Circuits</td>
      <td>https://coursera.org/learn/tinkering-circuits</td>
      <td>4.7</td>
      <td>Beginner Level</td>
      <td>['Personal Development', 'Personal Development']</td>
    </tr>
    <tr>
      <th>147</th>
      <td>147</td>
      <td>Improving Communication Skills</td>
      <td>https://coursera.org/learn/wharton-communicati...</td>
      <td>4.6</td>
      <td>Beginner Level</td>
      <td>['Personal Development', 'Personal Development']</td>
    </tr>
    <tr>
      <th>216</th>
      <td>216</td>
      <td>Studying at Japanese Universities</td>
      <td>https://coursera.org/learn/study-in-japan</td>
      <td>4.8</td>
      <td>Beginner Level</td>
      <td>['Personal Development', 'Personal Development']</td>
    </tr>
  </tbody>
</table>
</div>




```python
accounting = df[df['Tags'].str.contains("Accounting")|df['Name'].str.contains("Accounting")]
print(accounting.shape)
accounting.head()
```

    (7, 6)
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>Name</th>
      <th>Url</th>
      <th>Rating</th>
      <th>Difficulty</th>
      <th>Tags</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>61</th>
      <td>61</td>
      <td>Managerial Accounting Fundamentals</td>
      <td>https://coursera.org/learn/uva-darden-manageri...</td>
      <td>4.8</td>
      <td>Beginner Level</td>
      <td>['Business', 'Finance']</td>
    </tr>
    <tr>
      <th>262</th>
      <td>262</td>
      <td>Accounting and Finance for IT professionals</td>
      <td>https://coursera.org/learn/accounting-finance</td>
      <td>4.4</td>
      <td>Beginner Level</td>
      <td>['Business', 'Finance']</td>
    </tr>
    <tr>
      <th>264</th>
      <td>264</td>
      <td>Financial Accounting Fundamentals</td>
      <td>https://coursera.org/learn/uva-darden-financia...</td>
      <td>4.8</td>
      <td>Beginner Level</td>
      <td>['Business', 'Business Essentials']</td>
    </tr>
    <tr>
      <th>1763</th>
      <td>1763</td>
      <td>Accounting for Business Decision Making: Strat...</td>
      <td>https://coursera.org/learn/business-assessment</td>
      <td>4.1</td>
      <td>Beginner Level</td>
      <td>['Business', 'Business Essentials']</td>
    </tr>
    <tr>
      <th>2537</th>
      <td>2537</td>
      <td>Accounting: Principles of Financial Accounting</td>
      <td>https://coursera.org/learn/financial-accounting</td>
      <td>4.8</td>
      <td>Beginner Level</td>
      <td>['Business', 'Finance']</td>
    </tr>
  </tbody>
</table>
</div>




```python
business = df[df['Tags'].str.contains("Business")|df['Name'].str.contains("Business")]
print(business.shape)
business.head()
```

    (331, 6)
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>Name</th>
      <th>Url</th>
      <th>Rating</th>
      <th>Difficulty</th>
      <th>Tags</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>(Successful Presentation) العرض التقديمي الناجح</td>
      <td>https://coursera.org/learn/presentation-skills-ar</td>
      <td>4.9</td>
      <td>Beginner Level</td>
      <td>['Business', 'Business Essentials']</td>
    </tr>
    <tr>
      <th>34</th>
      <td>34</td>
      <td>Digital Product Management: Modern Fundamentals</td>
      <td>https://coursera.org/learn/uva-darden-digital-...</td>
      <td>4.7</td>
      <td>Beginner Level</td>
      <td>['Business', 'Leadership and Management']</td>
    </tr>
    <tr>
      <th>35</th>
      <td>35</td>
      <td>Digital Competition in Financial Services</td>
      <td>https://coursera.org/learn/digital-competition...</td>
      <td>4.6</td>
      <td>Beginner Level</td>
      <td>['Business', 'Business Strategy']</td>
    </tr>
    <tr>
      <th>41</th>
      <td>41</td>
      <td>Identifying Social Entrepreneurship Opportunities</td>
      <td>https://coursera.org/learn/social-entrepreneur...</td>
      <td>4.6</td>
      <td>Beginner Level</td>
      <td>['Business', 'Entrepreneurship']</td>
    </tr>
    <tr>
      <th>52</th>
      <td>52</td>
      <td>Compra programática de medios: Publicidad onli...</td>
      <td>https://coursera.org/learn/compra-programatica</td>
      <td>4.7</td>
      <td>Beginner Level</td>
      <td>['Business', 'Marketing']</td>
    </tr>
  </tbody>
</table>
</div>




```python
finance = df[df['Tags'].str.contains("Financ")|df['Name'].str.contains("Financ")] 
print(finance.shape)
finance.head()
```

    (54, 6)
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>Name</th>
      <th>Url</th>
      <th>Rating</th>
      <th>Difficulty</th>
      <th>Tags</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>35</th>
      <td>35</td>
      <td>Digital Competition in Financial Services</td>
      <td>https://coursera.org/learn/digital-competition...</td>
      <td>4.6</td>
      <td>Beginner Level</td>
      <td>['Business', 'Business Strategy']</td>
    </tr>
    <tr>
      <th>61</th>
      <td>61</td>
      <td>Managerial Accounting Fundamentals</td>
      <td>https://coursera.org/learn/uva-darden-manageri...</td>
      <td>4.8</td>
      <td>Beginner Level</td>
      <td>['Business', 'Finance']</td>
    </tr>
    <tr>
      <th>113</th>
      <td>113</td>
      <td>Valuation for Startups Using Multiple Approach</td>
      <td>https://coursera.org/learn/valuation-multiples</td>
      <td>4.2</td>
      <td>Beginner Level</td>
      <td>['Business', 'Finance']</td>
    </tr>
    <tr>
      <th>116</th>
      <td>116</td>
      <td>Administração Financeira</td>
      <td>https://coursera.org/learn/administracao-finan...</td>
      <td>4.8</td>
      <td>Beginner Level</td>
      <td>['Business', 'Finance']</td>
    </tr>
    <tr>
      <th>175</th>
      <td>175</td>
      <td>Financial Analysis for Startups</td>
      <td>https://coursera.org/learn/financial-ratios</td>
      <td>4.4</td>
      <td>Beginner Level</td>
      <td>['Business', 'Finance']</td>
    </tr>
  </tbody>
</table>
</div>




```python
adult = df[df['Tags'].str.contains("Adult")|df['Name'].str.contains("Adult")] 
print(adult.shape)
adult.head()
```

    (1, 6)
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>Name</th>
      <th>Url</th>
      <th>Rating</th>
      <th>Difficulty</th>
      <th>Tags</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>313</th>
      <td>313</td>
      <td>Financial Planning for Young Adults</td>
      <td>https://coursera.org/learn/financial-planning</td>
      <td>4.5</td>
      <td>Beginner Level</td>
      <td>['Business', 'Finance']</td>
    </tr>
  </tbody>
</table>
</div>




```python
education = df[df['Tags'].str.contains("Education")|df['Name'].str.contains("Education")] 
print(education.shape)
education.head()
```

    (49, 6)
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>Name</th>
      <th>Url</th>
      <th>Rating</th>
      <th>Difficulty</th>
      <th>Tags</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>47</th>
      <td>47</td>
      <td>Evaluación educativa del y para el aprendizaje...</td>
      <td>https://coursera.org/learn/evaluacion-educativa</td>
      <td>4.8</td>
      <td>Beginner Level</td>
      <td>['Social Sciences', 'Education']</td>
    </tr>
    <tr>
      <th>130</th>
      <td>130</td>
      <td>Powerful Tools for Teaching and Learning: Web ...</td>
      <td>https://coursera.org/learn/teaching-learning-t...</td>
      <td>4.6</td>
      <td>Beginner Level</td>
      <td>['Social Sciences', 'Education']</td>
    </tr>
    <tr>
      <th>171</th>
      <td>171</td>
      <td>Learning, Knowledge, and Human Development</td>
      <td>https://coursera.org/learn/learning-knowledge-...</td>
      <td>4</td>
      <td>Beginner Level</td>
      <td>['Social Sciences', 'Education']</td>
    </tr>
    <tr>
      <th>194</th>
      <td>194</td>
      <td>Gestión estratégica de Escuelas en Contextos R...</td>
      <td>https://coursera.org/learn/gestion-estrategica...</td>
      <td>4.9</td>
      <td>Beginner Level</td>
      <td>['Social Sciences', 'Education']</td>
    </tr>
    <tr>
      <th>211</th>
      <td>211</td>
      <td>Design Thinking for the Greater Good: Innovati...</td>
      <td>https://coursera.org/learn/uva-darden-design-t...</td>
      <td>4.5</td>
      <td>Beginner Level</td>
      <td>['Social Sciences', 'Education']</td>
    </tr>
  </tbody>
</table>
</div>




```python
tax = df[df['Tags'].str.contains("Tax")|df['Name'].str.contains("Tax")] 
tax.head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>Name</th>
      <th>Url</th>
      <th>Rating</th>
      <th>Difficulty</th>
      <th>Tags</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>



### * key result: no tax entry level courses with ratings on Coursera


```python
invest = df[df['Tags'].str.contains("investment")|df['Name'].str.contains("investment")] 
print(invest.shape)
invest.head()
# when filter through investing/invest, also no applicable results
```

    (0, 6)
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>Name</th>
      <th>Url</th>
      <th>Rating</th>
      <th>Difficulty</th>
      <th>Tags</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>



### * Key result: no entry level investment courses


```python
credit = df[df['Tags'].str.contains("credit")|df['Name'].str.contains("credit")] 
print(credit.shape)
credit.head()
```

    (0, 6)
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>Name</th>
      <th>Url</th>
      <th>Rating</th>
      <th>Difficulty</th>
      <th>Tags</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>



### * Key result: no entry level credit cards intro courses


```python
insurance = df[df['Tags'].str.contains("insurance")|df['Name'].str.contains("insurance")] 
print(insurance.shape)
insurance.head()
```

    (0, 6)
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>Name</th>
      <th>Url</th>
      <th>Rating</th>
      <th>Difficulty</th>
      <th>Tags</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>



### * Key result: no entry level insurance intro courses


```python
career = df[df['Tags'].str.contains("career")|df['Name'].str.contains("career")] 
print(career.shape)
career.head()
```

    (0, 6)
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>Name</th>
      <th>Url</th>
      <th>Rating</th>
      <th>Difficulty</th>
      <th>Tags</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>




```python
print(pd.to_numeric(finance["Rating"]).mean())
print(pd.to_numeric(business["Rating"]).mean())
print(pd.to_numeric(accounting["Rating"]).mean())
print(pd.to_numeric(adult["Rating"]).mean())
print(pd.to_numeric(education["Rating"]).mean())
print(pd.to_numeric(personal["Rating"]).mean())

```

    4.633333333333334
    4.646827794561932
    4.557142857142857
    4.5
    4.6387755102040815
    4.587931034482759
    


```python
plotdata = pd.DataFrame(
    {"Ratings": [4.633333333333334,
4.646827794561932,
4.557142857142857,
4.5,
4.6387755102040815,
4.587931034482759]}, 
    index=["finance", "business", "accounting", "adult", "education","personal"])
# Plot a bar chart
plotdata.plot(kind="bar")
plt.xticks(rotation=30, horizontalalignment="center")
plt.title("Rating Comparisons for Relevant Courses at Beginners Level")
plt.xlabel("Key Words")
plt.ylabel("Ratings")
```




    Text(0, 0.5, 'Ratings')




![png](output_18_1.png)



```python
print(finance.shape[0])
print(business.shape[0])
print(accounting.shape[0])
print(adult.shape[0])
print(education.shape[0])
print(personal.shape[0])

```

    54
    331
    7
    1
    49
    58
    


```python
plotdata = pd.DataFrame(
    {"# of courses": [54,
331,
7,
1,
49,
58]}, 
    index=["finance", "business", "accounting", "adult", "education","personal"])
# Plot a bar chart
plotdata.plot(kind="bar")
plt.xticks(rotation=30, horizontalalignment="center")
plt.title("Number of Revelant Courses at Beginners Level")
plt.xlabel("Key Words")
plt.ylabel("Number of Courses provided")
```




    Text(0, 0.5, 'Number of Courses provided')




![png](output_20_1.png)



```python
plotdata = pd.DataFrame(
    {"Popularity": [46,
331,
7,
1,
49,
58]}, 
    index=["finance", "business", "accounting", "adult", "education","personal"])
# Plot a bar chart
plotdata.plot.pie(y='Popularity', figsize=(6, 6))
plt.title("Number of Revelant Courses at Beginners Level")
```




    Text(0.5, 1.0, 'Number of Revelant Courses at Beginners Level')




![png](output_21_1.png)


### * Potential Relevant Courses (for non-professionals and personal):

#### business related:
- Financial Accounting Fundamentals
- **Accounting for Decision Making**
#### career:
- **Improving Communication Skills**
- **Blended Learning: Personalizing Education for Students**
- **How to Write a Resume (Project-Centered Course)**
- **Creative Thinking: Techniques and Tools for Success**
- Career Brand Development and Self-Coaching
- Success
- Speaking to inform:  Discussing complex ideas with clear explanations and dynamic slides
- IBM IT Assessment: Identifying the Right Career for You!
- Essentials for English Speeches and Presentations
- Personality Types at Work
- Effective Communication in the Globalised Workplace - The Capstone
#### personal life:
- **Financial Planning for Young Adults**
- **International Travel Preparation, Safety, & Wellness**
- Empowering Yourself in a Post-Truth World
- What does it mean to identify as Transgender or Gender Non-Conforming (TGNC)?
- The Arts and Science of Relationships: Understanding Human Needs
- Healing with the Arts

### summaries:
1. No/very few tax/insurance/credit cards/investing related courses at entry level for personals on Coursera, however, users might actually look for courses such as investment capstone course designed for academic use to learn first timer investment knowledge
2. Business related coursework is most popular, Onomy can design courses such as how to help grow your business designed for adults.

### thoughts: 
1. Currently even Onomy create more adulting courses, users might not pay for them. There are no sustainable and continuing video resources for them to stick to the platform, especially compared to other video platforms. If I watch one videa such as how to invest as a first-timer, then I would understand the skills and might never use the website again. So perhaps new strategies besides *adulting* might need to be considered.Potential topics:
   - When become adults, people would definitely face job finding, which is probably the most important thing. We could provide a series of relevant coursework such as how to polish resume, what resources you can turn to in locating a job, how to network, how to pitch yourself,etc.Then we can provide afterward courses such as how to enhance work efficiency, communication skills, how to find part time work for additional revenues etc. This series of coursework might attract continuing subscribers. 
       - a lot of people outside schools/in schools that doesn't provide a series of job preparation will be in need of a holistic tutorial on how to find for better jobs; currently a lot of people are under unemployment, the strategies to locate new jobs would be in need
   - Since we target people who focus on personal finance, we can probably develop more courses on how to manage your account, what other asset classes are available to invest(fixed income, bonds, etc). Courses can act as a continuing consultant, which tells users what to learn at a step by step over a series of courses.
   - For health insurance part on Onomy, it might be good to create videos as a Valuator for different insurance companies. List out benefits, costs, and other characteristics of different insurance plan that allow users to know at once the differentiation between different companies, and get a better idea of what to choose
    
2. There are many guides online other than videos to learn for adulting stuff. For instance, for the first time investing konwledge, users might google and simply read guides(such as https://www.nerdwallet.com/article/investing/how-to-start-investing) that might actually provide more resources all at onece. So we might need to consider provide comprehensive lessons from entry level to more in-depth levels as a competitive advantage.



```python

```
