package org.janelia.render.service.util;

import org.janelia.alignment.Utils;

import javax.imageio.stream.ImageOutputStream;
import javax.imageio.stream.MemoryCacheImageOutputStream;
import javax.ws.rs.WebApplicationException;
import javax.ws.rs.core.StreamingOutput;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.io.OutputStream;

/**
 * Wrapper for {@link java.awt.image.BufferedImage} instances that need to be
 * streamed as the response for a JAX-RS API request.
 * Uses {@link org.janelia.alignment.Utils#writeImage} to do the real work.
 *
 * @author Eric Trautman
 */
public class BufferedImageStreamingOutput implements StreamingOutput {

    private BufferedImage targetImage;
    private String format;
    private boolean convertToGray;
    private float quality;

    public BufferedImageStreamingOutput(BufferedImage targetImage,
                                        String format,
                                        boolean convertToGray,
                                        float quality) {
        this.targetImage = targetImage;
        this.format = format;
        this.convertToGray = convertToGray;
        this.quality = quality;
    }

    @Override
    public void write(OutputStream output)
            throws IOException, WebApplicationException {

        final ImageOutputStream imageOutputStream = new MemoryCacheImageOutputStream(output);
        Utils.writeImage(targetImage, format, convertToGray, quality, imageOutputStream);
    }

}