//
// Shinobi - Tensorflow Plugin
// Copyright (C) 2016-2025 Moe Alam, moeiscool
// Copyright (C) 2020 Levent Koch, dermodmaster
//
// # Donate
//
// If you like what I am doing here and want me to continue please consider donating :)
// PayPal : paypal@m03.ca
//
// Base Init >>
var fs = require('fs');
var config = require('./conf.json')
var dotenv = require('dotenv').config()
var s
try {
    s = require('../pluginBase.js')(__dirname, config)
} catch (err) {
    console.log(err)
    try {
        s = require('./pluginBase.js')(__dirname, config)
    } catch (err) {
        console.log(err)
        return console.log(config.plug, 'Plugin start has failed. pluginBase.js was not found.')
    }
}
 
 
var ready = false;
const spawn = require('child_process').spawn;
var child = null
function respawn() {
 
    console.log("respawned python",(new Date()))
    const theChild = spawn('python3', ['-u', 'detect_image.py']);
 
    var lastStatusLog = new Date();
 
    theChild.on('exit', () => {
         child = respawn();
    });
 
    theChild.stdout.on('data', function (data) {
        var rawString = data.toString('utf8');
        if (new Date() - lastStatusLog > 5000) {
            lastStatusLog = new Date();
            console.log(rawString, new Date());
        }
        var messages = rawString.split('\n')
        messages.forEach((message) => {
            if (message === "") return;
            var obj = JSON.parse(message)
            if (obj.type === "error") {
                console.log("Script got error: " + message.data, new Date());
                throw message.data;
            }
 
            if (obj.type === "info" && obj.data === "ready") {
                console.log("set ready true")
                ready = true;
            } else {
                if (obj.type !== "data" && obj.type !== "info") {
                    throw "Unexpected message: " + rawString;
                }
            }
        })
    })
    return theChild
}
 
 
 
 
// Base Init />>
child = respawn();
 
const emptyDataObject = { data: [], type: undefined, time: 0 };
 
async function process(buffer, type) {
    const startTime = new Date();
    if (!ready) {
        return emptyDataObject;
    }
    ready = false;
    child.stdin.write(buffer.toString('base64') + '\n');
 
    var message = null;
    await new Promise(resolve => {
        child.stdout.once('data', (data) => {
            var rawString = data.toString('utf8').split("\n")[0];
            try {
                message = JSON.parse(rawString)
            }
            catch (e) {
                message = { data: [] };
            }
            resolve();
        });
    })
    const data = message.data;
    ready = true;
    return {
        data: data,
        type: type,
        time: new Date() - startTime
    }
}
 
 
s.detectObject = function (buffer, d, tx, frameLocation, callback) {
    process(buffer).then((resp) => {
        var results = resp.data
        //console.log(resp.time)
        if (Array.isArray(results) && results[0]) {
            var mats = []
            results.forEach(function (v) {
                mats.push({
                    x: v.bbox[0],
                    y: v.bbox[1],
                    width: v.bbox[2],
                    height: v.bbox[3],
                    tag: v.class,
                    confidence: v.score,
                })
            })
            var isObjectDetectionSeparate = d.mon.detector_pam === '1' && d.mon.detector_use_detect_object === '1'
            var width = parseFloat(isObjectDetectionSeparate && d.mon.detector_scale_y_object ? d.mon.detector_scale_y_object : d.mon.detector_scale_y)
            var height = parseFloat(isObjectDetectionSeparate && d.mon.detector_scale_x_object ? d.mon.detector_scale_x_object : d.mon.detector_scale_x)
            tx({
                f: 'trigger',
                id: d.id,
                ke: d.ke,
                details: {
                    plug: config.plug,
                    name: 'Tensorflow',
                    reason: 'object',
                    matrices: mats,
                    imgHeight: width,
                    imgWidth: height,
                    time: resp.time
                }
            })
        }
        callback()
    })
}
