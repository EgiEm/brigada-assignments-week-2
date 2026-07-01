const fs = require('fs');
const path = require('path');

function buildDataset(csvPath = 'intents.csv') {
    const fullPath = path.resolve(__dirname, csvPath);
    if (!fs.existsSync(fullPath)) {
        console.error(`Dataset file ${fullPath} not found.`);
        process.exit(1);
    }

    const content = fs.readFileSync(fullPath, 'utf8');
    const lines = content.split('\n');
    
    const rows = [];
    const seenTexts = new Set();
    const duplicates = [];
    const classCounts = {};

    let headerSkipped = false;

    for (let i = 0; i < lines.length; i++) {
        const line = lines[i].trim();
        if (!line) continue;
        if (!headerSkipped) {
            headerSkipped = true;
            continue;
        }

        const lastCommaIndex = line.lastIndexOf(',');
        if (lastCommaIndex === -1) continue;
        let text = line.substring(0, lastCommaIndex).trim();
        const label = line.substring(lastCommaIndex + 1).trim();
        
        if (text.startsWith('"') && text.endsWith('"')) {
            text = text.substring(1, text.length - 1).trim();
        }

        if (seenTexts.has(text)) {
            duplicates.push({ text, line: i + 1 });
        }
        seenTexts.add(text);

        classCounts[label] = (classCounts[label] || 0) + 1;
        rows.push({ text, label });
    }

    console.log("=== DATASET BUILD REPORT ===");
    console.log(`Total Rows: ${rows.length}`);
    console.log(`Unique Sentences: ${seenTexts.size}`);

    if (duplicates.length > 0) {
        console.log("\nWARNING: Leakage detected! Duplicate entries found:");
        duplicates.forEach(dup => {
            console.log(` - '${dup.text}' (Line ${dup.line})`);
        });
    } else {
        console.log("\nClean Check: Dataset is leakage-free (0 duplicates).");
    }

    console.log("\nClass Distribution:");
    const sortedLabels = Object.keys(classCounts).sort();
    sortedLabels.forEach(label => {
        const count = classCounts[label];
        console.log(` - ${label.padEnd(18)} : ${count} rows`);
    });

    const counts = Object.values(classCounts);
    const minCount = Math.min(...counts);
    const maxCount = Math.max(...counts);

    if (minCount === maxCount && sortedLabels.length === 6) {
        console.log("\nBalance Check: PERFECTLY BALANCED across all 6 intents.");
    } else if (maxCount - minCount <= 1) {
        console.log("\nBalance Check: Roughly balanced (difference <= 1).");
    } else {
        console.log("\nWARNING: Dataset is imbalanced!");
    }
}

buildDataset('intents.csv');
