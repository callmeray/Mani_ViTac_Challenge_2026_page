# 字体更换指南

## 当前字体设置

当前使用的是 **Inter** 字体（Google Fonts）。

## 更换字体的方法

### 方法 1: 使用 Google Fonts（推荐）

#### 步骤 1: 修改 HTML 文件中的字体链接

在 `index.html` 的 `<head>` 部分，找到这一行：
```html
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
```

替换为您想要的字体，例如：

**Roboto（现代无衬线）:**
```html
<link href="https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700;900&display=swap" rel="stylesheet">
```

**Poppins（圆润现代）:**
```html
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
```

**Montserrat（几何风格）:**
```html
<link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
```

**Open Sans（清晰易读）:**
```html
<link href="https://fonts.googleapis.com/css2?family=Open+Sans:wght@300;400;600;700;800&display=swap" rel="stylesheet">
```

**Noto Sans SC（中文字体，适合中文内容）:**
```html
<link href="https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@300;400;500;700;900&display=swap" rel="stylesheet">
```

#### 步骤 2: 修改 CSS 文件中的字体名称

在 `styles.css` 文件中，找到 `body` 样式：
```css
body {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}
```

将 `'Inter'` 替换为您选择的字体名称，例如：
- `'Roboto'`
- `'Poppins'`
- `'Montserrat'`
- `'Open Sans'`
- `'Noto Sans SC'`

### 方法 2: 使用系统字体（无需加载外部字体）

如果您想使用系统自带的字体，可以：

1. **删除** `index.html` 中的 Google Fonts 链接（第 8-10 行）
2. 在 `styles.css` 中修改为：
```css
body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Microsoft YaHei', 'PingFang SC', 'Hiragino Sans GB', sans-serif;
}
```

### 方法 3: 使用自定义字体文件

如果您有自己的字体文件（.woff2, .woff, .ttf 等）：

1. 将字体文件放在项目文件夹中（建议创建 `fonts` 文件夹）
2. 在 `styles.css` 中添加 `@font-face`：
```css
@font-face {
    font-family: 'YourFontName';
    src: url('fonts/yourfont.woff2') format('woff2'),
         url('fonts/yourfont.woff') format('woff');
    font-weight: normal;
    font-style: normal;
}
```
3. 在 `body` 中使用：
```css
body {
    font-family: 'YourFontName', sans-serif;
}
```

## 推荐的字体组合

### 科技/现代风格
- **Inter**（当前使用）- 简洁现代
- **Roboto** - Google 设计，清晰易读
- **Montserrat** - 几何风格，适合标题

### 专业/学术风格
- **Open Sans** - 清晰专业
- **Lato** - 友好易读
- **Source Sans Pro** - Adobe 设计

### 中文友好
- **Noto Sans SC** - Google 中文字体
- **PingFang SC** - 苹果系统字体（仅 macOS/iOS）
- **Microsoft YaHei** - 微软雅黑（Windows）

## 快速更换示例

如果您想快速更换为 **Roboto** 字体：

1. 修改 `index.html` 第 10 行：
```html
<link href="https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700;900&display=swap" rel="stylesheet">
```

2. 修改 `styles.css` 第 27 行：
```css
font-family: 'Roboto', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
```

完成！刷新页面即可看到新字体效果。
