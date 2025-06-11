# Computer-Price-Predictor

### Problem Statement:

 1 Introduction
 The 15.csv Turkish E-Commerce Desktop Computer dataset provides comprehensive information
 on desktop computers and hardware components available on a popular e-commerce platform in Turkey.
 This dataset serves as a valuable resource for researchers, data scientists, and analysts interested in
 hardware specifications, pricing trends, and e-commerce insights. By analyzing this dataset, one can gain
 a deeper understanding of market dynamics, pricing patterns, and consumer preferences in the desktop
 computer segment.


 2 Dataset Description
 The dataset consists of 28 attributes capturing key technical specifications and pricing information.
 The primary attributes include:
 2.1 General Information
 • Brand (Marka)– Manufacturer of the computer (e.g., XASER, HP, Dell).
 • Price (Fiyat)– Selling price of the desktop (in Turkish Lira- TL).
 • Country of Manufacture (Men¸sei)– Country where the product was manufactured.
 2.2 Processor and Memory Specifications
 • Processor Type (˙ I¸slemci Tipi)– Brand and model of the CPU (e.g., Intel Core i5, AMD Ryzen
 7).
 • Base Clock Speed (Temel ˙ I¸slemci Hızı)– Processor’s base speed in GHz.
 • Processor Frequency (˙ I¸slemci Frekansı)– Maximum speed of the processor (above 3.00 GHz,
 etc.).
 • RAM (Sistem Belle˘gi)– Installed system memory (e.g., 8 GB, 16 GB).
 • Expandable Maximum Memory (Arttırılabilir Azami Bellek)– Maximum RAM capacity.
 • SSD Capacity (SSD Kapasitesi)– Storage capacity of the SSD (e.g., 256 GB, 512 GB).
 2.3 Graphics and Display Features
 • Graphics Card (Ekran Kartı)– Model of the GPU.
 • Graphics Card Memory (Ekran Kartı Kapasitesi)– VRAM capacity (e.g., 4 GB, 8 GB).
 • Graphics Memory Type (Ekran Kartı Bellek Tipi)– Type of GPU memory (e.g., DDR3,
 GDDR5).
 • Graphics Card Type (Ekran Kartı Tipi)– Dedicated or integrated GPU.
 • Monitor Size (Ekran Boyutu)– Screen size in inches (e.g., 24”, 27”).
 • Screen Refresh Rate (Ekran Yenileme Hızı)– Display refresh rate (e.g., 75 Hz, 165 Hz).
 • Panel Type (Panel Tipi)– Monitor panel technology (VA, IPS, TN).
 1
2.4 Connectivity and Physical Attributes
 • Connections (Ba˘glantılar)– Available ports such as HDMI, DisplayPort.
 • Weight (Cihaz A˘gırlı˘gı)– Total weight of the desktop computer.

 
 3 Tasks and Requirements
 To analyze and extract meaningful insights from the dataset, the following tasks need to be performed:
 3.1 Data Exploration and Preprocessing
 • Load and clean the dataset, handling missing values and inconsistent formatting.
 • Normalize numerical attributes such as price, processor speed, and RAM.
 • Convert categorical variables (e.g., brand, processor type) into structured formats for analysis.
 • Identify outliers in pricing and technical specifications.
 3.2 Market and Pricing Analysis
 • Analyze pricing trends across different brands and hardware configurations.
 • Examine the relationship between technical specifications (e.g., RAM, SSD size, processor type)
 and pricing.
 • Identify popular brands and models based on price distribution.
 • Predict pricing trends using linear regression or time-series forecasting models.
 3.3 Hardware Component Trends
 • Investigate the most common processor and graphics card configurations.
 • Compare performance attributes of different brands and price segments.
 • Detect emerging trends in RAM and SSD capacity over time.
 3.4 Visualization and Reporting
 • Generate distribution plots to visualize price variation among brands.
 • Create heatmaps to analyze correlations between system specifications and pricing.
 • Develop bar charts showcasing popular configurations and feature trends.
