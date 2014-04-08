/**
 * License: GPL
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License 2
 * as published by the Free Software Foundation.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.
 */
package org.janelia.alignment;

import ij.ImagePlus;
import ij.io.Opener;

import java.awt.image.BufferedImage;
import java.awt.image.RenderedImage;
import java.io.File;
import java.io.IOException;
import java.net.URL;
import java.util.ArrayList;

import javax.imageio.ImageIO;

import mpicbg.models.AbstractModel;
import mpicbg.models.AffineModel2D;
import mpicbg.models.CoordinateTransform;
import mpicbg.models.HomographyModel2D;
import mpicbg.models.NotEnoughDataPointsException;
import mpicbg.models.Point;
import mpicbg.models.PointMatch;
import mpicbg.models.RigidModel2D;
import mpicbg.models.SimilarityModel2D;
import mpicbg.models.SpringMesh;
import mpicbg.models.Tile;
import mpicbg.models.TranslationModel2D;

/**
 * 
 *
 * @author Stephan Saalfeld <saalfeld@janelia.hhmi.org>, Seymour Knowles-Barley
 */
public class Utils
{
	private Utils() {}
	
	final static public class Triple< A, B, C >
	{
		final public A a;
		final public B b;
		final public C c;

		Triple( final A a, final B b, final C c )
		{
			this.a = a;
			this.b = b;
			this.c = c;
		}
	}
	
	/**
	 * Get a model from an integer specifier
	 */
	final static public AbstractModel< ? > createModel( final int modelIndex )
	{
		switch ( modelIndex )
		{
		case 0:
			return new TranslationModel2D();
		case 1:
			return new RigidModel2D();
		case 2:
			return new SimilarityModel2D();
		case 3:
			return new AffineModel2D();
		case 4:
			return new HomographyModel2D();
		default:
			return null;
		}
	}
	
	/**
	 * Get a tile from an integer specifier
	 */
	final static public Tile< ? > createTile( final int modelIndex )
	{
		switch ( modelIndex )
		{
		case 0:
			return (Tile< ? >) new Tile< TranslationModel2D >( new TranslationModel2D() );
		case 1:
			return (Tile< ? >) new Tile< RigidModel2D >( new RigidModel2D() );
		case 2:
			return (Tile< ? >) new Tile< SimilarityModel2D >( new SimilarityModel2D() );
		case 3:
			return (Tile< ? >) new Tile< AffineModel2D >( new AffineModel2D() );
		case 4:
			return (Tile< ? >) new Tile< HomographyModel2D >( new HomographyModel2D() );
		default:
			return null;
		}
	}
	
	/**
	 * Generate a spring mesh from image dimensions and spring mesh parameters.
	 */
	public static SpringMesh getMesh( int imWidth, int imHeight, float layerScale,
			int resolutionSpringMesh, float stiffnessSpringMesh, float dampSpringMesh, float maxStretchSpringMesh )
	{
		final int meshWidth = ( int )Math.ceil( imWidth * layerScale );
		final int meshHeight = ( int )Math.ceil( imHeight * layerScale );
		
		final SpringMesh mesh = new SpringMesh(
						resolutionSpringMesh,
						meshWidth,
						meshHeight,
						stiffnessSpringMesh,
						maxStretchSpringMesh * layerScale,
						dampSpringMesh );
		
		return mesh;
	}
	
	/**
	 * Save an image using ImageIO.
	 * 
	 * @param image
	 * @param path
	 * @param format
	 */
	final static public boolean saveImage( final RenderedImage image, final String path, final String format )
	{
		try
		{
			final File file = new File( path );
			ImageIO.write( image, format, file );
			return true;
		}
		catch ( final IOException e )
		{
			return false;
		}
	}
	
	/**
	 * Open an ImagePlus from a file.
	 * 
	 * @param pathString
	 * @return
	 */
	final static public ImagePlus openImagePlus( final String pathString )
	{
		final ImagePlus imp = new Opener().openImage( pathString );
		return imp;
	}
	
	/**
	 * Open an ImagePlus from a URL
	 * 
	 * @param urlString
	 * @return
	 */
	final static public ImagePlus openImagePlusUrl( final String urlString )
	{
		final ImagePlus imp = new Opener().openURL( imageJUrl( urlString ) );
		return imp;
	}
	
	/**
	 * Open an Image from a URL.  Try ImageIO first, then ImageJ.
	 * 
	 * @param urlString
	 * @return
	 */
	final static public BufferedImage openImageUrl( final String urlString )
	{
		BufferedImage image;
		try
		{
			final URL url = new URL( urlString );
			final BufferedImage imageTemp = ImageIO.read( url );
			
			/* This gymnastic is necessary to get reproducible gray
			 * values, just opening a JPG or PNG, even when saved by
			 * ImageIO, and grabbing its pixels results in gray values
			 * with a non-matching gamma transfer function, I cannot tell
			 * why... */
		    image = new BufferedImage( imageTemp.getWidth(), imageTemp.getHeight(), BufferedImage.TYPE_INT_ARGB );
			image.createGraphics().drawImage( imageTemp, 0, 0, null );
		}
		catch ( final Exception e )
		{
			try
			{
				final ImagePlus imp = openImagePlusUrl( urlString );
				if ( imp != null )
				{
					image = imp.getBufferedImage();
				}
				else image = null;
			}
			catch ( final Exception f )
			{
				image = null;
			}
		}
		return image;
	}
	
	
	/**
	 * Open an Image from a file.  Try ImageIO first, then ImageJ.
	 * 
	 * @param urlString
	 * @return
	 */
	final static public BufferedImage openImage( final String path )
	{
		BufferedImage image = null;
		try
		{
			final File file = new File( path );
			if ( file.exists() )
			{
				final BufferedImage jpg = ImageIO.read( file );
				
				/* This gymnastic is necessary to get reproducible gray
				 * values, just opening a JPG or PNG, even when saved by
				 * ImageIO, and grabbing its pixels results in gray values
				 * with a non-matching gamma transfer function, I cannot tell
				 * why... */
			    image = new BufferedImage( jpg.getWidth(), jpg.getHeight(), BufferedImage.TYPE_INT_ARGB );
				image.createGraphics().drawImage( jpg, 0, 0, null );
			}
		}
		catch ( final Exception e )
		{
			try
			{
				final ImagePlus imp = openImagePlus( path );
				if ( imp != null )
				{
					image = imp.getBufferedImage();
				}
				else image = null;
			}
			catch ( final Exception f )
			{
				image = null;
			}
		}
		return image;
	}
	
	
	/**
	 * If a URL starts with "file:", replace "file:" with "" because ImageJ wouldn't understand it otherwise
	 * @return
	 */
	final static private String imageJUrl( final String urlString )
	{
		return urlString.replace( "^file:", "" );
	}
	
	
	/**
	 * Combine a 0x??rgb int[] raster and an unsigned byte[] alpha channel into
	 * a 0xargb int[] raster.  The operation is perfomed in place on the int[]
	 * raster.
	 */
	final static public void combineARGB( final int[] rgb, final byte[] a )
	{
		for ( int i = 0; i < rgb.length; ++i )
		{
			rgb[ i ] &= 0x00ffffff;
			rgb[ i ] |= ( a[ i ] & 0xff ) << 24;
		}
	}
	
	
	/**
	 * Sample the average scaling of a given {@link CoordinateTransform} by transferring
	 * a set of point samples using the {@link CoordinateTransform} and then
	 * least-squares fitting a {@link SimilarityModel2D} to it.
	 * 
	 * @param ct
	 * @param width of the samples set
	 * @param height of the samples set
	 * @param dx spacing between samples
	 * 
	 * @return average scale factor
	 */
	final static public double sampleAverageScale( final CoordinateTransform ct, final int width, final int height, final double dx )
	{
		final ArrayList< PointMatch > samples = new ArrayList< PointMatch >();
		for ( float y = 0; y < height; y += dx )
		{
			for ( float x = 0; x < width; x += dx )
			{
				final Point p = new Point( new float[]{ x, y } );
				p.apply( ct );
				samples.add( new PointMatch( p, p ) );
			}
		}
		final SimilarityModel2D model = new SimilarityModel2D();
		try
		{
			model.fit( samples );
		}
		catch ( final NotEnoughDataPointsException e )
		{
			e.printStackTrace( System.err );
			return 1;
		}
		final double[] data = new double[ 6 ];
		model.toArray( data );
//		return 1;
		return Math.sqrt( data[ 0 ] * data[ 0 ] + data[ 1 ] * data[ 1 ] );
	}
	
	
	final static public int bestMipmapLevel( final double scale )
	{
		int invScale = ( int )( 1.0 / scale );
		int scaleLevel = 0;
		while ( invScale > 1 )
		{
			invScale >>= 1;
			++scaleLevel;
		}
		return scaleLevel;
	}
	
	/**
	 * Create an affine transformation that compensates for both scale and
	 * pixel shift of a mipmap level that was generated by top-left pixel
	 * averaging.
	 * 
	 * @param scaleLevel
	 * @return
	 */
	final static AffineModel2D createScaleLevelTransform( final int scaleLevel )
	{
		final AffineModel2D a = new AffineModel2D();
		final int scale = 1 << scaleLevel;
		final float t = ( scale - 1 ) * 0.5f;
		a.set( scale, 0, 0, scale, t, t );
		return a;
	}
}