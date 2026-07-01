// Import the built-in 'fs' (File System) module to read files from disk
const fs = require('fs');
// Import the built-in 'path' module to resolve relative/absolute file paths
const path = require('path');

// Main function to load and validate the dataset, similar to the Python lab implementation
function buildDataset(csvPath = 'intents.csv') {
    // Resolve the absolute file path of intents.csv based on the current directory
    const fullPath = path.resolve(__dirname, csvPath);
    // Check if the dataset CSV file exists; exit with error if it does not
    if (!fs.existsSync(fullPath)) {
        console.error(`Dataset file ${fullPath} not found.`);
        process.exit(1);
    }

    // Read the contents of the CSV file synchronously as a UTF-8 string
    const content = fs.readFileSync(fullPath, 'utf8');
    // Split the file contents by line breaks to get an array of lines
    const lines = content.split('\n');
    
    const rows = [];            // Array to store parsed {text, label} objects
    const seenTexts = new Set();  // Set to track unique sentences to detect duplicates (leakage)
    const duplicates = [];      // Array to collect duplicate lines for reporting
    const classCounts = {};    // Object to track occurrences of each intent label

    let headerSkipped = false;  // Flag to skip the CSV header line during processing

    // Loop through each line of the CSV file
    for (let i = 0; i < lines.length; i++) {
        const line = lines[i].trim();
        // Skip empty lines in the CSV file
        if (!line) continue;
        // Skip the very first line (text,label header)
        if (!headerSkipped) {
            headerSkipped = true;
            continue;
        }

        // Find the index of the last comma on this line
        // We use lastIndexOf so that text containing commas (like German sentences) 
        // will not be split incorrectly, since the label is always the last field.
        const lastCommaIndex = line.lastIndexOf(',');
        if (lastCommaIndex === -1) continue; // Skip malformed lines with no commas
        
        // Extract text and label fields based on the last comma's position
        let text = line.substring(0, lastCommaIndex).trim();
        const label = line.substring(lastCommaIndex + 1).trim();
        
        // If the text is wrapped in double quotes, strip the surrounding quotes
        if (text.startsWith('"') && text.endsWith('"')) {
            text = text.substring(1, text.length - 1).trim();
        }

        // Check for data leakage: if we've already seen this text, record it as a duplicate
        if (seenTexts.has(text)) {
            duplicates.push({ text, line: i + 1 });
        }
        seenTexts.add(text); // Add the text to our unique set

        // Track counts per class/label and add to rows array
        classCounts[label] = (classCounts[label] || 0) + 1;
        rows.push({ text, label });
    }

    // Print the dataset verification report
    console.log("=== DATASET BUILD REPORT ===");
    console.log(`Total Rows: ${rows.length}`);
    console.log(`Unique Sentences: ${seenTexts.size}`);

    // If duplicate texts are present, warn the user and list the offending entries
    if (duplicates.length > 0) {
        console.log("\nWARNING: Leakage detected! Duplicate entries found:");
        duplicates.forEach(dup => {
            console.log(` - '${dup.text}' (Line ${dup.line})`);
        });
    } else {
        console.log("\nClean Check: Dataset is leakage-free (0 duplicates).");
    }

    // Display the counts of each intent class sorted alphabetically
    console.log("\nClass Distribution:");
    const sortedLabels = Object.keys(classCounts).sort();
    sortedLabels.forEach(label => {
        const count = classCounts[label];
        console.log(` - ${label.padEnd(18)} : ${count} rows`);
    });

    // Check if the dataset classes are balanced
    const counts = Object.values(classCounts);
    const minCount = Math.min(...counts);
    const maxCount = Math.max(...counts);

    // If counts are equal and all 6 classes are represented, the dataset is perfectly balanced
    if (minCount === maxCount && sortedLabels.length === 6) {
        console.log("\nBalance Check: PERFECTLY BALANCED across all 6 intents.");
    } else if (maxCount - minCount <= 1) {
        console.log("\nBalance Check: Roughly balanced (difference <= 1).");
    } else {
        console.log("\nWARNING: Dataset is imbalanced!");
    }
}

// Execute the dataset verification process
buildDataset('intents.csv');
