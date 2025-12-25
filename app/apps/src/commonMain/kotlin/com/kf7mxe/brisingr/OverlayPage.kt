//package com.kf7mxe.brisingr
//
//package com.kf7mxe.layerpainter.pages
//
//import com.lightningkite.kiteui.FileReference
//import com.lightningkite.kiteui.Routable
//import com.lightningkite.kiteui.mimeType
//import com.lightningkite.kiteui.navigation.Page
//import com.lightningkite.kiteui.requestFile
//import com.lightningkite.kiteui.views.ViewWriter
//import com.lightningkite.kiteui.views.card
//import com.lightningkite.kiteui.views.centered
//import com.lightningkite.kiteui.views.direct.BottomSheetState
//import com.lightningkite.kiteui.views.direct.button
//import com.lightningkite.kiteui.views.direct.col
//import com.lightningkite.kiteui.views.direct.coordinatorFrame
//import com.lightningkite.kiteui.views.direct.onClick
//import com.lightningkite.kiteui.views.direct.row
//import com.lightningkite.kiteui.views.direct.text
//import com.lightningkite.kiteui.views.expanding
//import com.lightningkite.kiteui.views.l2.coordinatorFrame
//import com.lightningkite.readable.Constant
//import com.lightningkite.readable.Property
//import com.lightningkite.readable.Readable
//
//@Routable("/front-lit-create-edit")
//class FrontLitCreateEditPage : Page {
//    override val title: Readable<String> get() = Constant("Front Lit")
//    val selectedImage = Property<FileReference?>(null)
//    val generatedImagePreview = Property<FileReference?>(null)
//
//
//    override fun ViewWriter.render(): ViewModifiable {
//        return expanding - col {
//            expanding - coordinatorFrame {
//                coordinatorFrame = this
//
//                centered - button {
//                    text("Add image")
//                    onClick {
//                        val image = context.requestFile(listOf("image/png", "image/jpeg"))
//                        if (image?.mimeType()?.contains("image") != true) throw Exception("Not an image")
//                        selectedImage.set(image)
//                        processImageUsingFilamentPainter(image)
//
//                    }
//                }
//                preview(selectedImage, generatedImagePreview, BottomSheetState.COLLAPSED)
//            }
//            row {
//                expanding - card - button {
//                    centered - expanding - text("Save and Export")
//                }
//                expanding - card - button {
//                    centered - expanding - text("Save")
//                }
//            }
////            }
//        }
//    }
//}
//
//fun processImageUsingFilamentPainter(image: FileReference) {
////    val getComputeEngine = getComputeFunction(HeightFunction.GREYSCALE_MAX)
////    val glImage = GLImage()
////    println("getComputeEngine: $getComputeEngine")
//
//
//    // dispose of GL stuff first TODO()
//
//    val selectedHeightFunctions = HeightFunction.GREYSCALE_MAX
//    when (selectedHeightFunctions) {
//        HeightFunction.NEAREST -> {
//            // run compute shader
//            val vertexShader = createNearestVertexShader()
//            vertexShader.setSourceOfCodeOfShader()
//            vertexShader.compileShader()
//
//            val fragmentShader = createNearestFragmentShader()
//            fragmentShader.setSourceOfCodeOfShader()
//            fragmentShader.compileShader()
//            val program = createGLProgram()
//            program.attachShader(vertexShader)
//            program.attachShader(fragmentShader)
//            program.linkProgram()
//
//
//            //compute
//            program.useProgram()
//
//            val glFrameBuffer = GLFrameBuffer()
//            val glTexture = GLTexture()
//            glTexture.bindTexture()
//            glFrameBuffer.bindFrameBuffer()
//            gl.bindTexture(gl.TEXTURE_2D, outputTexture);
//            gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, textureWidth, textureHeight, 0, gl.RGBA, gl.FLOAT, null);
//            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
//            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
//            gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, outputTexture, 0);
//
//            let filaments : Filament [] = config.paint.filaments;
//
//
//
//            const colours =[];
//            const heights =[];
//            const opacities =[];
//            let heightRange =[];
//
//            for (let i = 0; i < filaments.length; i++) {
//                let filament = filaments [i];
//                colours.push(filament.colour[0]);
//                colours.push(filament.colour[1]);
//                colours.push(filament.colour[2]);
//                heights.push(filament.endHeight);
//                opacities.push(filament.opacity);
//            }
//
//            heightRange = [config.paint.startHeight, config.paint.endHeight, config.paint.increment];
//
//            if (heights.length == 0) {
//                return new Float32Array ();
//            }
//
//            this.uploadComputeData(
//                colours,
//                heights,
//                opacities,
//                heightRange,
//                image
//            );
//
//            gl.viewport(0, 0, textureWidth, textureHeight);
//
//            gl.drawArrays(gl.TRIANGLES, 0, 6);
//
//            const outputData = new Float32Array(textureWidth * textureHeight * 4);
//            gl.readPixels(0, 0, textureWidth, textureHeight, gl.RGBA, gl.FLOAT, outputData);
//
//            gl.bindFramebuffer(gl.FRAMEBUFFER, null);
//            gl.bindTexture(gl.TEXTURE_2D, null);
//            gl.deleteFramebuffer(framebuffer);
//            gl.deleteTexture(outputTexture);
//
//            return outputData;
//            debugDisplayDataOutput(computedResult, image.width, image.height);
//            debugDisplayHTMLImage(image);
//
//
//
//            TODO()
//        }
//
//        HeightFunction.GREYSCALE_MAX -> {
//            TODO()
//        }
//
//        HeightFunction.GREYSCALE_LUMINANCE -> {
//            TODO()
//        }
//    }
//
//}
//
//
//export function debugDisplayHTMLImage(image: HTMLImageElement) {
//    const canvas : HTMLCanvasElement | null = document.getElementById('canvas-source') as HTMLCanvasElement | null;
//
//    if (!canvas) {
//        console.error('Canvas element with ID "canvas-source" not found.');
//        return;
//    }
//
//    const ctx = canvas . getContext ('2d');
//
//    if (!ctx) {
//        console.error('Could not get 2D rendering context.');
//        return;
//    }
//
//    canvas.width = image.width;
//    canvas.height = image.height;
//
//    ctx.drawImage(image, 0, 0);
//}
//
