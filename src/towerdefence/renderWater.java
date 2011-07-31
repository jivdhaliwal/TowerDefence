package towerdefence;

import org.newdawn.slick.Animation;
import org.newdawn.slick.GameContainer;
import org.newdawn.slick.Graphics;
import org.newdawn.slick.Image;
import org.newdawn.slick.SlickException;
import org.newdawn.slick.SpriteSheet;
import org.newdawn.slick.state.StateBasedGame;

import towerdefence.engine.ResourceManager;

/**
 *
 * Renders the animated water background
 * TODO Ability to change water direction
 * 
 * @author Jiv Dhaliwal <jivdhaliwal@gmail.com>
 */
public final class RenderWater {

    SpriteSheet waterSheet;
    Image[] waterFrames;
    Animation waterAnimation;

    int tw, th;

    public RenderWater(int tw, int th) throws SlickException {

        this.tw = tw;
        this.th = th;

        Image water = ResourceManager.getInstance().getImage("WATER");
        waterSheet = new SpriteSheet(water, 32, 32);
        waterFrames = new Image[8];
        setRandomWater();

    }

    public void setRandomWater() {
        
        java.util.Random random = new java.util.Random();
        int xWater = random.nextInt(3);
        int yWater = random.nextInt(3);
        if(xWater == 1 || yWater == 1) {
            xWater = 0;
            yWater = 0;
        }
        for(int x=0;x<waterFrames.length;x++) {
            waterFrames[x]=waterSheet.getSprite(x*3+xWater, yWater);
        }
        waterAnimation = new Animation(waterFrames, 100,true);
    }

    public void render(GameContainer container, StateBasedGame game, Graphics g) throws SlickException {
        for(int i=0;i<tw;i++) {
            for(int j=0;j<th;j++) {
                waterAnimation.draw(i*32,j*32);
            }
        }

    }

}
