# AI for Landlords – Data & Integration Documentation

## Overview

The “AI for Landlords” platform provides three core AI-powered services:
- **Rent Prediction**
- **Tenant Screening**
- **Predictive Maintenance**

These services are accessed through an interactive assistant (chatbot) interface, powered by OpenAI and custom machine learning models. Below, we describe the data requirements, expected formats, and the workflow for each AI module.

---

## 1. Rent Prediction AI

### **Purpose**
Estimates the optimal rent for a property based on its features and location.

### **Required Data Fields**
| Field                        | Type    | Example                | Description                                 |
|------------------------------|---------|------------------------|---------------------------------------------|
| address                      | string  | "123 Main St"          | Property address or unique identifier       |
| subdistrict_code             | string  | "UB8"                  | Area or postal code                         |
| BEDROOMS                     | int     | 2                      | Number of bedrooms                          |
| BATHROOMS                    | int     | 1                      | Number of bathrooms                         |
| SIZE                         | float   | 700.0                  | Size in square feet                         |
| PROPERTY TYPE                | string  | "flat"                 | e.g., flat, house, apartment                |
| avg_distance_to_nearest_station | float | 0.5                    | Average distance to nearest station (miles) |
| nearest_station_count        | int     | 2                      | Number of nearby stations                   |

### **Data Format Example**
```json
{
  "address": "123 Main St",
  "subdistrict_code": "UB8",
  "BEDROOMS": 2,
  "BATHROOMS": 1,
  "SIZE": 700.0,
  "PROPERTY TYPE": "flat",
  "avg_distance_to_nearest_station": 0.5,
  "nearest_station_count": 2
}
```

### **How to Provide Data**
- Data can be provided via the chatbot in natural language or as structured input.
- The assistant will prompt for missing fields if not all are provided.

---

## 2. Tenant Screening AI

### **Purpose**
Assesses the suitability of a tenant based on financial and background information.

### **Required Data Fields**
| Field            | Type    | Example      | Description                        |
|------------------|---------|--------------|------------------------------------|
| credit_score     | int     | 700          | Applicant’s credit score           |
| income           | float   | 3500.0       | Monthly income                     |
| rent             | float   | 1200.0       | Monthly rent for the property      |
| employment_status| string  | "employed"   | Employment status                  |
| eviction_record  | bool    | false        | Any prior eviction? (true/false)   |

### **Data Format Example**
```json
{
  "credit_score": 700,
  "income": 3500.0,
  "rent": 1200.0,
  "employment_status": "employed",
  "eviction_record": false
}
```

### **How to Provide Data**
- Data can be provided in conversation or as a list.
- The assistant will ask for any missing information.

---

## 3. Predictive Maintenance AI

### **Purpose**
Predicts and alerts for potential maintenance issues across properties.

### **Required Data Fields**
| Field             | Type    | Example      | Description                                 |
|-------------------|---------|--------------|---------------------------------------------|
| address           | string  | "123 Main St"| Property address                            |
| construction_type | string  | "brick"      | Type of construction                        |
| age_years         | int     | 20           | Age of property in years                    |
| hvac_age          | int     | 5            | Age of HVAC system                          |
| plumbing_age      | int     | 10           | Age of plumbing system                      |
| roof_age          | int     | 8            | Age of roof                                 |
| total_incidents   | int     | 3            | Total maintenance incidents                 |
| urgent_incidents  | int     | 1            | Number of urgent incidents                  |
| open_issues       | int     | 2            | Number of unresolved issues                 |

### **Data Format Example**
```json
{
  "address": "123 Main St",
  "construction_type": "brick",
  "age_years": 20,
  "hvac_age": 5,
  "plumbing_age": 10,
  "roof_age": 8,
  "total_incidents": 3,
  "urgent_incidents": 1,
  "open_issues": 2
}
```

### **How to Provide Data**
- Data can be uploaded in bulk in the database.
- The assistant can summarize and alert for all properties with urgent needs.

---

## 4. Data Flow & Integration Diagram

<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns:xlink="http://www.w3.org/1999/xlink" aria-roledescription="flowchart-v2" role="graphics-document document" viewBox="0 0 1029.6000366210938 350" style="max-width: 1029.6000366210938px;" class="flowchart" xmlns="http://www.w3.org/2000/svg" width="100%" id="mermaid-svg-85"><style>#mermaid-svg-85{font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:16px;fill:#333;}@keyframes edge-animation-frame{from{stroke-dashoffset:0;}}@keyframes dash{to{stroke-dashoffset:0;}}#mermaid-svg-85 .edge-animation-slow{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 50s linear infinite;stroke-linecap:round;}#mermaid-svg-85 .edge-animation-fast{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 20s linear infinite;stroke-linecap:round;}#mermaid-svg-85 .error-icon{fill:#552222;}#mermaid-svg-85 .error-text{fill:#552222;stroke:#552222;}#mermaid-svg-85 .edge-thickness-normal{stroke-width:1px;}#mermaid-svg-85 .edge-thickness-thick{stroke-width:3.5px;}#mermaid-svg-85 .edge-pattern-solid{stroke-dasharray:0;}#mermaid-svg-85 .edge-thickness-invisible{stroke-width:0;fill:none;}#mermaid-svg-85 .edge-pattern-dashed{stroke-dasharray:3;}#mermaid-svg-85 .edge-pattern-dotted{stroke-dasharray:2;}#mermaid-svg-85 .marker{fill:#333333;stroke:#333333;}#mermaid-svg-85 .marker.cross{stroke:#333333;}#mermaid-svg-85 svg{font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:16px;}#mermaid-svg-85 p{margin:0;}#mermaid-svg-85 .label{font-family:"trebuchet ms",verdana,arial,sans-serif;color:#333;}#mermaid-svg-85 .cluster-label text{fill:#333;}#mermaid-svg-85 .cluster-label span{color:#333;}#mermaid-svg-85 .cluster-label span p{background-color:transparent;}#mermaid-svg-85 .label text,#mermaid-svg-85 span{fill:#333;color:#333;}#mermaid-svg-85 .node rect,#mermaid-svg-85 .node circle,#mermaid-svg-85 .node ellipse,#mermaid-svg-85 .node polygon,#mermaid-svg-85 .node path{fill:#ECECFF;stroke:#9370DB;stroke-width:1px;}#mermaid-svg-85 .rough-node .label text,#mermaid-svg-85 .node .label text,#mermaid-svg-85 .image-shape .label,#mermaid-svg-85 .icon-shape .label{text-anchor:middle;}#mermaid-svg-85 .node .katex path{fill:#000;stroke:#000;stroke-width:1px;}#mermaid-svg-85 .rough-node .label,#mermaid-svg-85 .node .label,#mermaid-svg-85 .image-shape .label,#mermaid-svg-85 .icon-shape .label{text-align:center;}#mermaid-svg-85 .node.clickable{cursor:pointer;}#mermaid-svg-85 .root .anchor path{fill:#333333!important;stroke-width:0;stroke:#333333;}#mermaid-svg-85 .arrowheadPath{fill:#333333;}#mermaid-svg-85 .edgePath .path{stroke:#333333;stroke-width:2.0px;}#mermaid-svg-85 .flowchart-link{stroke:#333333;fill:none;}#mermaid-svg-85 .edgeLabel{background-color:rgba(232,232,232, 0.8);text-align:center;}#mermaid-svg-85 .edgeLabel p{background-color:rgba(232,232,232, 0.8);}#mermaid-svg-85 .edgeLabel rect{opacity:0.5;background-color:rgba(232,232,232, 0.8);fill:rgba(232,232,232, 0.8);}#mermaid-svg-85 .labelBkg{background-color:rgba(232, 232, 232, 0.5);}#mermaid-svg-85 .cluster rect{fill:#ffffde;stroke:#aaaa33;stroke-width:1px;}#mermaid-svg-85 .cluster text{fill:#333;}#mermaid-svg-85 .cluster span{color:#333;}#mermaid-svg-85 div.mermaidTooltip{position:absolute;text-align:center;max-width:200px;padding:2px;font-family:"trebuchet ms",verdana,arial,sans-serif;font-size:12px;background:hsl(80, 100%, 96.2745098039%);border:1px solid #aaaa33;border-radius:2px;pointer-events:none;z-index:100;}#mermaid-svg-85 .flowchartTitleText{text-anchor:middle;font-size:18px;fill:#333;}#mermaid-svg-85 rect.text{fill:none;stroke-width:0;}#mermaid-svg-85 .icon-shape,#mermaid-svg-85 .image-shape{background-color:rgba(232,232,232, 0.8);text-align:center;}#mermaid-svg-85 .icon-shape p,#mermaid-svg-85 .image-shape p{background-color:rgba(232,232,232, 0.8);padding:2px;}#mermaid-svg-85 .icon-shape rect,#mermaid-svg-85 .image-shape rect{opacity:0.5;background-color:rgba(232,232,232, 0.8);fill:rgba(232,232,232, 0.8);}#mermaid-svg-85 :root{--mermaid-font-family:"trebuchet ms",verdana,arial,sans-serif;}</style><g><marker orient="auto" markerHeight="8" markerWidth="8" markerUnits="userSpaceOnUse" refY="5" refX="5" viewBox="0 0 10 10" class="marker flowchart-v2" id="mermaid-svg-85_flowchart-v2-pointEnd"><path style="stroke-width: 1; stroke-dasharray: 1, 0;" class="arrowMarkerPath" d="M 0 0 L 10 5 L 0 10 z"></path></marker><marker orient="auto" markerHeight="8" markerWidth="8" markerUnits="userSpaceOnUse" refY="5" refX="4.5" viewBox="0 0 10 10" class="marker flowchart-v2" id="mermaid-svg-85_flowchart-v2-pointStart"><path style="stroke-width: 1; stroke-dasharray: 1, 0;" class="arrowMarkerPath" d="M 0 5 L 10 10 L 10 0 z"></path></marker><marker orient="auto" markerHeight="11" markerWidth="11" markerUnits="userSpaceOnUse" refY="5" refX="11" viewBox="0 0 10 10" class="marker flowchart-v2" id="mermaid-svg-85_flowchart-v2-circleEnd"><circle style="stroke-width: 1; stroke-dasharray: 1, 0;" class="arrowMarkerPath" r="5" cy="5" cx="5"></circle></marker><marker orient="auto" markerHeight="11" markerWidth="11" markerUnits="userSpaceOnUse" refY="5" refX="-1" viewBox="0 0 10 10" class="marker flowchart-v2" id="mermaid-svg-85_flowchart-v2-circleStart"><circle style="stroke-width: 1; stroke-dasharray: 1, 0;" class="arrowMarkerPath" r="5" cy="5" cx="5"></circle></marker><marker orient="auto" markerHeight="11" markerWidth="11" markerUnits="userSpaceOnUse" refY="5.2" refX="12" viewBox="0 0 11 11" class="marker cross flowchart-v2" id="mermaid-svg-85_flowchart-v2-crossEnd"><path style="stroke-width: 2; stroke-dasharray: 1, 0;" class="arrowMarkerPath" d="M 1,1 l 9,9 M 10,1 l -9,9"></path></marker><marker orient="auto" markerHeight="11" markerWidth="11" markerUnits="userSpaceOnUse" refY="5.2" refX="-1" viewBox="0 0 11 11" class="marker cross flowchart-v2" id="mermaid-svg-85_flowchart-v2-crossStart"><path style="stroke-width: 2; stroke-dasharray: 1, 0;" class="arrowMarkerPath" d="M 1,1 l 9,9 M 10,1 l -9,9"></path></marker><g class="root"><g class="clusters"></g><g class="edgePaths"><path marker-end="url(#mermaid-svg-85_flowchart-v2-pointEnd)" style="" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" id="L_A_B_0" d="M480.1,62L470.85,68.167C461.6,74.333,443.1,86.667,442.545,98.63C441.991,110.594,459.381,122.187,468.077,127.984L476.772,133.781"></path><path marker-end="url(#mermaid-svg-85_flowchart-v2-pointEnd)" style="" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" id="L_B_C1_0" d="M449.4,173.949L391.9,182.79C334.4,191.632,219.4,209.316,170.39,225.876C121.38,242.436,138.36,257.873,146.85,265.591L155.34,273.309"></path><path marker-end="url(#mermaid-svg-85_flowchart-v2-pointEnd)" style="" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" id="L_B_C2_0" d="M486.091,190L478.209,196.167C470.327,202.333,454.564,214.667,454.983,228.546C455.403,242.426,472.006,257.852,480.308,265.564L488.609,273.277"></path><path marker-end="url(#mermaid-svg-85_flowchart-v2-pointEnd)" style="" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" id="L_B_C3_0" d="M591.8,181.569L620.833,189.141C649.867,196.713,707.933,211.856,743.997,225.173C780.06,238.49,794.119,249.979,801.149,255.724L808.179,261.469"></path><path marker-end="url(#mermaid-svg-85_flowchart-v2-pointEnd)" style="" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" id="L_C1_B_0" d="M217.7,276L226.683,267.833C235.667,259.667,253.633,243.333,291.604,227.716C329.575,212.099,387.551,197.197,416.538,189.747L445.526,182.296"></path><path marker-end="url(#mermaid-svg-85_flowchart-v2-pointEnd)" style="" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" id="L_C2_B_0" d="M549.661,276L558.45,267.833C567.24,259.667,584.82,243.333,586.253,229.411C587.687,215.488,572.973,203.977,565.616,198.221L558.26,192.465"></path><path marker-end="url(#mermaid-svg-85_flowchart-v2-pointEnd)" style="" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" id="L_C3_B_0" d="M906.724,264L914.27,257.833C921.816,251.667,936.908,239.333,885.08,224.358C833.252,209.383,714.504,191.767,655.131,182.958L595.757,174.15"></path><path marker-end="url(#mermaid-svg-85_flowchart-v2-pointEnd)" style="" class="edge-thickness-normal edge-pattern-solid edge-thickness-normal edge-pattern-solid flowchart-link" id="L_B_A_0" d="M561.1,136L570.35,129.833C579.6,123.667,598.1,111.333,598.655,99.37C599.209,87.406,581.819,75.813,573.124,70.016L564.428,64.219"></path></g><g class="edgeLabels"><g transform="translate(424.6000003814697, 99)" class="edgeLabel"><g transform="translate(-82.80000305175781, -12)" class="label"><foreignObject height="24" width="165.60000610351562"><div style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;" class="labelBkg" xmlns="http://www.w3.org/1999/xhtml"><span class="edgeLabel"><p>Provides Property Data</p></span></div></foreignObject></g></g><g transform="translate(104.4000015258789, 227)" class="edgeLabel"><g transform="translate(-96.4000015258789, -12)" class="label"><foreignObject height="24" width="192.8000030517578"><div style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;" class="labelBkg" xmlns="http://www.w3.org/1999/xhtml"><span class="edgeLabel"><p>Extracts &amp; Validates Fields</p></span></div></foreignObject></g></g><g transform="translate(438.8000030517578, 227)" class="edgeLabel"><g transform="translate(-96.4000015258789, -12)" class="label"><foreignObject height="24" width="192.8000030517578"><div style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;" class="labelBkg" xmlns="http://www.w3.org/1999/xhtml"><span class="edgeLabel"><p>Extracts &amp; Validates Fields</p></span></div></foreignObject></g></g><g transform="translate(766.0000076293945, 227)" class="edgeLabel"><g transform="translate(-96.4000015258789, -12)" class="label"><foreignObject height="24" width="192.8000030517578"><div style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;" class="labelBkg" xmlns="http://www.w3.org/1999/xhtml"><span class="edgeLabel"><p>Extracts &amp; Validates Fields</p></span></div></foreignObject></g></g><g transform="translate(271.60000228881836, 227)" class="edgeLabel"><g transform="translate(-50.79999923706055, -12)" class="label"><foreignObject height="24" width="101.5999984741211"><div style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;" class="labelBkg" xmlns="http://www.w3.org/1999/xhtml"><span class="edgeLabel"><p>Rent Estimate</p></span></div></foreignObject></g></g><g transform="translate(602.4000053405762, 227)" class="edgeLabel"><g transform="translate(-47.20000076293945, -12)" class="label"><foreignObject height="24" width="94.4000015258789"><div style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;" class="labelBkg" xmlns="http://www.w3.org/1999/xhtml"><span class="edgeLabel"><p>Tenant Score</p></span></div></foreignObject></g></g><g transform="translate(952.0000076293945, 227)" class="edgeLabel"><g transform="translate(-69.5999984741211, -12)" class="label"><foreignObject height="24" width="139.1999969482422"><div style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;" class="labelBkg" xmlns="http://www.w3.org/1999/xhtml"><span class="edgeLabel"><p>Maintenance Alerts</p></span></div></foreignObject></g></g><g transform="translate(616.6000080108643, 99)" class="edgeLabel"><g transform="translate(-89.20000457763672, -12)" class="label"><foreignObject height="24" width="178.40000915527344"><div style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;" class="labelBkg" xmlns="http://www.w3.org/1999/xhtml"><span class="edgeLabel"><p>Provides Results &amp; Alerts</p></span></div></foreignObject></g></g></g><g class="nodes"><g transform="translate(520.600004196167, 35)" id="flowchart-A-0" class="node default"><rect height="54" width="92" y="-27" x="-46" style="fill:#f9f !important;stroke:#333 !important" class="basic label-container"></rect><g transform="translate(-16, -12)" style="" class="label"><rect></rect><foreignObject height="24" width="32"><div style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;" xmlns="http://www.w3.org/1999/xhtml"><span class="nodeLabel"><p>User</p></span></div></foreignObject></g></g><g transform="translate(520.600004196167, 163)" id="flowchart-B-1" class="node default"><rect height="54" width="142.4000015258789" y="-27" x="-71.20000076293945" style="fill:#bbf !important;stroke:#333 !important" class="basic label-container"></rect><g transform="translate(-41.20000076293945, -12)" style="" class="label"><rect></rect><foreignObject height="24" width="82.4000015258789"><div style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;" xmlns="http://www.w3.org/1999/xhtml"><span class="nodeLabel"><p>AI Assistant</p></span></div></foreignObject></g></g><g transform="translate(188.00000190734863, 303)" id="flowchart-C1-3" class="node default"><rect height="54" width="219.1999969482422" y="-27" x="-109.5999984741211" style="fill:#8f8 !important;stroke:#333 !important" class="basic label-container"></rect><g transform="translate(-79.5999984741211, -12)" style="" class="label"><rect></rect><foreignObject height="24" width="159.1999969482422"><div style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;" xmlns="http://www.w3.org/1999/xhtml"><span class="nodeLabel"><p>Rent Prediction Model</p></span></div></foreignObject></g></g><g transform="translate(520.600004196167, 303)" id="flowchart-C2-5" class="node default"><rect height="54" width="232.8000030517578" y="-27" x="-116.4000015258789" style="fill:#8f8 !important;stroke:#333 !important" class="basic label-container"></rect><g transform="translate(-86.4000015258789, -12)" style="" class="label"><rect></rect><foreignObject height="24" width="172.8000030517578"><div style="display: table-cell; white-space: nowrap; line-height: 1.5; max-width: 200px; text-align: center;" xmlns="http://www.w3.org/1999/xhtml"><span class="nodeLabel"><p>Tenant Screening Model</p></span></div></foreignObject></g></g><g transform="translate(859.0000076293945, 303)" id="flowchart-C3-7" class="node default"><rect height="78" width="260" y="-39" x="-130" style="fill:#8f8 !important;stroke:#333 !important" class="basic label-container"></rect><g transform="translate(-100, -24)" style="" class="label"><rect></rect><foreignObject height="48" width="200"><div style="display: table; white-space: break-spaces; line-height: 1.5; max-width: 200px; text-align: center; width: 200px;" xmlns="http://www.w3.org/1999/xhtml"><span class="nodeLabel"><p>Maintenance Prediction Model</p></span></div></foreignObject></g></g></g></g></g></svg>

---

## 5. How the User Should Provide Data

- **Via Chatbot:**  
  The User can type or paste property/tenant details in natural language. The assistant will extract the required fields and prompt for any missing information.
- **Via Bulk Upload (for maintenance):**  
  For predictive maintenance, the User can provide a spreadsheet or CSV with the required columns.
- **Data Quality:**  
  - Ensure all required fields are present and accurate for best results.
  - For categorical fields (like property type), use standard terms (e.g., “flat”, “house”, “apartment”).

---

## 6. Interactive Layer

- The assistant uses OpenAI’s language models to interact with users, extract data, and route requests to the correct AI module.
- The system is robust to natural language and can handle both structured and unstructured input.

---

## 7. Summary

- **Data can be provided casually, but try to keep the information as clear as possible.**
- The assistant will guide users to supply any missing information.
- The system is designed to be user-friendly and flexible for both individual and bulk property management.

---

**If you have any questions about data formats or integration, please contact the project team.**

---
