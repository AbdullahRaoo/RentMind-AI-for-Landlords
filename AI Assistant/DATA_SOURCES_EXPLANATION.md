## 📊 **Data Sources for Area Comparison Feature**

### 🗂️ **Primary Data Sources:**

**1. Original Rental Listings Data:**
- **File:** `/Rent Pricing AI/data/rent_ads_rightmove_extended.csv`
- **Source:** Rightmove (UK's largest property portal)
- **Size:** 3,574 rental listings
- **Content:** Raw rental ads with full property details

**2. Processed Rental Data:**
- **File:** `/Rent Pricing AI/data/cleaned_rent_data.csv`
- **Source:** Cleaned version of Rightmove data
- **Size:** 3,479 properties (after cleaning)
- **Content:** Standardized rental data for ML model

**3. Address Mapping:**
- **File:** `/Rent Pricing AI/address_map_human.json`
- **Source:** Processed address mapping
- **Size:** 2,880 unique addresses
- **Content:** Human-readable addresses mapped to numeric IDs

---

### 🔄 **Data Flow for Area Comparison:**

```
User Query: "Which area has higher rent, Battersea or Clapham?"
                              ⬇️
1. **Extract Area Names** (chatbot_integration.py)
   - Uses regex patterns to identify: ["Battersea", "Clapham"]
                              ⬇️
2. **Search Address Mapping** (address_map_human.json)
   - Find all addresses containing "battersea": 
     * "Alder House, 2 Electric Boulevard, Battersea, SW11" → ID: 126
     * "Battersea Park Road, Battersea, SW11" → ID: 257
     * "Falcon Wharf, Battersea Riverside" → ID: 986
     * etc.
                              ⬇️
3. **Filter Rental Data** (cleaned_rent_data.csv)
   - Match address IDs to rental records
   - Extract: rent, bedrooms, bathrooms, property_type
                              ⬇️
4. **Calculate Statistics**
   - Average rent, median, min/max ranges
   - Property counts, sample listings
                              ⬇️
5. **Generate Comparison Report**
   - Formatted response with statistics and insights
```

---

### 📋 **Sample Data Records:**

**From rent_ads_rightmove_extended.csv:**
```csv
address,rent,BEDROOMS,BATHROOMS,PROPERTY_TYPE
"Alder House, 2 Electric Boulevard, Battersea, SW11",8880,3.0,3.0,Flat
"Battersea Park Road, London, SW11",2149,1.0,1.0,Apartment
"Emu Road, Battersea, SW8",2300,2.0,1.0,House
```

**After Processing (cleaned_rent_data.csv):**
```csv
address,subdistrict_code,BEDROOMS,BATHROOMS,SIZE,PROPERTY_TYPE,rent
126,SW11,3.0,3.0,,Flat,8880
257,SW11,1.0,1.0,,Apartment,2149
954,SW8,2.0,1.0,,House,2300
```

---

### 🎯 **Data Quality & Coverage:**

- **Geographic Coverage:** London and surrounding areas
- **Data Freshness:** Real rental listings from Rightmove
- **Property Types:** Flats, apartments, houses, studios
- **Price Range:** £150 - £78,000 per month
- **Total Properties:** 3,479 cleaned listings

---

### 💡 **How the Comparison Works:**

1. **User asks:** "Battersea vs Clapham rent?"
2. **System finds:** 11 Battersea properties, 5 Clapham properties  
3. **Calculates averages:** Battersea £3,156/month, Clapham £1,789/month
4. **Provides insight:** "Battersea is 76.4% more expensive"
5. **Shows samples:** Real property examples from each area

This ensures **100% data-driven responses** based on actual rental market data! 🏠📈
